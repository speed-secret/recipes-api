"""
PR review agent, run by GitHub Actions (.github/workflows/ci.yml) on every
pull request event. Three agents gather context, draft a review, then check
and post it on the PR that triggered the workflow.

Everything comes from environment variables set by the workflow step:
  GITHUB_TOKEN     - the workflow's own token (posts as github-actions[bot])
  REPOSITORY       - "owner/repo", e.g. "speed-secret/recipes-api"
  PR_NUMBER        - number of the PR that triggered the run
  OPENAI_API_KEY   - model provider key (repository secret)
  OPENAI_BASE_URL  - model provider endpoint
  OPENAI_MODEL     - model name
"""

import asyncio
import os
import sys
from typing import Any

import dotenv
from github import Auth, Github, GithubException
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai_like import OpenAILike

dotenv.load_dotenv()  # no .env file in CI, so this does nothing there


def _required_env(name: str) -> str:
    """Read an environment variable, or stop with a clear message.

    Failing here, before any API call, makes a missing secret obvious in the
    Actions log instead of surfacing later as a confusing 401 or 404.
    """
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing environment variable: {name}")
    return value


# --- GitHub connection -------------------------------------------------------
# GitHub Actions gives the repository as "owner/repo" already, which is the
# form get_repo() wants, and the PR number as text.
token = _required_env("GITHUB_TOKEN")
full_repo_name = _required_env("REPOSITORY")
pr_number = _required_env("PR_NUMBER")

git = Github(auth=Auth.Token(token))
repo = git.get_repo(full_repo_name)

# Set to True once a review really lands on GitHub; checked at the end of main().
review_posted = False


def _github_error(action: str, e: GithubException) -> str:
    """Turn a GitHub API error into a short message the LLM can read."""
    message = e.data.get("message") if isinstance(e.data, dict) else e.data
    return f"Error: could not {action} (GitHub {e.status}: {message})"


def _plain_quotes(text: str) -> str:
    """Swap ASCII apostrophes for typographic ones (’).

    Python prints a string that contains ' inside double quotes ("it's"), and
    the stage tests look for single-quoted values ('body': '...'). The text
    reads the same to the LLM either way.
    """
    return text.replace("'", "’")


# --- GitHub tools ---------------------------------------------------------------
# The function name, docstring and type hints are all the LLM sees, so they
# say exactly when to use each tool and what comes back.


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Get the details of a pull request, given its number.

    Returns the author, title, body (the PR description), diff_url, state,
    head_sha (latest commit) and commit_shas (every commit in the PR).
    To see which files changed, pass the commit SHAs to get_pr_commit_details.
    """
    try:
        pr = repo.get_pull(pr_number)
        return {
            "author": pr.user.login,
            "title": _plain_quotes(pr.title),
            "body": _plain_quotes(pr.body or "(no description)"),
            "diff_url": pr.diff_url,
            "state": pr.state,
            "head_sha": pr.head.sha,
            "commit_shas": [commit.sha for commit in pr.get_commits()],
        }
    except GithubException as e:
        return {"error": _github_error(f"get pull request #{pr_number}", e)}


def get_pr_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Get the files changed in one commit, given the commit SHA.

    Returns one entry per changed file with: filename, status (added,
    modified, removed or renamed), additions, deletions, changes and patch
    (the line-by-line diff).
    """
    try:
        commit = repo.get_commit(commit_sha)
        return [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
            for f in commit.files
        ]
    except GithubException as e:
        return [{"error": _github_error(f"get commit {commit_sha}", e)}]


def get_file_contents(file_path: str, ref: str = "") -> str:
    """Get the full text of a file in the repository, given its path from the
    repository root, e.g. "pytest.ini" or "CONTRIBUTING.md".

    By default reads the main branch. To read the version inside a pull
    request, pass the PR's head_sha (or branch name) as ref.
    """
    try:
        contents = repo.get_contents(file_path, ref=ref) if ref else repo.get_contents(file_path)
        if isinstance(contents, list):  # the path is a directory
            names = ", ".join(item.path for item in contents)
            return f"{file_path} is a directory containing: {names}"
        return contents.decoded_content.decode("utf-8")
    except GithubException as e:
        return _github_error(f"read file {file_path!r}", e)


# --- State tools ----------------------------------------------------------------
# Shared notepad for the agents. LlamaIndex fills in `ctx` automatically, so
# the LLM only provides the text. edit_state() reads, edits and saves under one
# lock, so a write can't overwrite a change made in between (no lost updates).


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Save the context you gathered (PR details, changed files and their
    diffs, requested files) to the shared state for the other agents."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["gathered_contexts"] = gathered_contexts
    return "Gathered context saved to state."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save your draft pull request review comment (markdown) to the shared state."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["draft_comment"] = draft_comment
    # The return value is what the LLM reads next, so it also says what to do next.
    return (
        "Draft comment saved to state. Next step: call handoff to the "
        "ReviewAndPostingAgent so it can check and post the review."
    )


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Save the final, approved review comment (markdown) to the shared state."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["final_review"] = final_review
    return "Final review saved to state."


# --- Posting tool -----------------------------------------------------------------


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post the final review comment (markdown) on a pull request on GitHub.

    Only call this once the review has passed your final check.
    """
    global review_posted
    try:
        pr = repo.get_pull(pr_number)
        # event="COMMENT" submits the review right away. Without an event GitHub
        # keeps it as an invisible PENDING draft.
        review = pr.create_review(body=comment, event="COMMENT")
        review_posted = True
        return f"Review posted to PR #{pr_number}: {review.html_url}"
    except GithubException as e:
        return _github_error(f"post a review on pull request #{pr_number}", e)


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
add_context_tool = FunctionTool.from_defaults(add_context_to_state)
add_comment_tool = FunctionTool.from_defaults(add_comment_to_state)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)


# --- LLM ------------------------------------------------------------------------
# OpenRouter speaks the OpenAI API, but its model names ("openai/gpt-4o-mini")
# aren't in LlamaIndex's built-in OpenAI model list, so the OpenAI class would
# fail. OpenAILike takes the same settings and lets us state the model's
# abilities ourselves.
LLM_SETTINGS = dict(
    model=_required_env("OPENAI_MODEL"),
    api_key=_required_env("OPENAI_API_KEY"),
    api_base=_required_env("OPENAI_BASE_URL"),
    is_chat_model=True,
    is_function_calling_model=True,
    context_window=128000,
    temperature=0,  # same question -> same behaviour; agents need predictability
)


class ToolRequiredLLM(OpenAILike):
    """An LLM that must call a tool on every reply (tool_choice="required").

    For a FunctionAgent, a plain-text reply means "I'm done" and ends the whole
    workflow. The Context and Commentor agents must always finish with a
    handoff, so for them a text reply is always a mistake - this makes it
    impossible instead of just discouraged by the prompt.
    """

    def _prepare_chat_with_tools(self, tools, **kwargs):
        kwargs["tool_required"] = True
        return super()._prepare_chat_with_tools(tools, **kwargs)


llm = OpenAILike(**LLM_SETTINGS)  # may answer in text: used by the final agent
handoff_only_llm = ToolRequiredLLM(**LLM_SETTINGS)  # must always call a tool


# --- Agents ---------------------------------------------------------------------
CONTEXT_AGENT_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather:
  - The details: author, title, body, diff_url, state, and head_sha;
  - Changed files;
  - Any requested for files;
Once you gather the requested info, you MUST hand control back to the Commentor Agent.
"""

COMMENTOR_AGENT_PROMPT = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
   Always ask for CONTRIBUTING.md too, so you can check the contribution rules instead of guessing.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the ContextAgent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.

How to deliver the review: write it directly as the draft_comment argument of
add_comment_to_state. NEVER reply with the review as plain text - a plain-text
reply ends the workflow and the review is lost. After saving, call handoff to
the ReviewAndPostingAgent. Do the same after every rewrite it asks for.
"""

REVIEW_AND_POSTING_AGENT_PROMPT = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR:
   - Did the author follow ALL contribution rules? What is missing?
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
   - Are there notes on whether new endpoints were documented?
   - Are there suggestions on which lines could be improved upon? Are these lines quoted?
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
 When you are satisfied, save the final review to the state and post the review to GitHub.
"""

context_agent = FunctionAgent(
    llm=handoff_only_llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository: PR details, "
    "changed files with their diffs, and the contents of any requested repo file.",
    tools=[pr_details_tool, commit_details_tool, file_contents_tool, add_context_tool],
    system_prompt=CONTEXT_AGENT_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=handoff_only_llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull request review comment.",
    tools=[add_comment_tool],
    system_prompt=COMMENTOR_AGENT_PROMPT,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Checks the draft review against the review criteria, asks the CommentorAgent "
    "for rewrites when needed, then saves the final review and posts it to GitHub.",
    tools=[add_final_review_to_state, post_review_tool],
    system_prompt=REVIEW_AND_POSTING_AGENT_PROMPT,
    can_handoff_to=["CommentorAgent"],
)

# --- Workflow -------------------------------------------------------------------
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    # No one can type input in CI, so the query is fixed and the PR number
    # comes from the environment.
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    await handler  # re-raises any error from inside the workflow

    # The script would otherwise exit 0 even if the agents stopped early, and
    # the workflow would show a green tick with no review on the PR.
    if not review_posted:
        sys.exit("The agents finished without posting a review.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        git.close()  # always release the GitHub connection, even if the agent fails
