from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.utils import timezone

from app.models import Recipe, RecipeRating
from app.serializers import RecipeRatingSerializer


@pytest.fixture
def recipe():
    return Recipe.objects.create(
        title="Tomato soup", time_minutes=20, price="4.50"
    )


@pytest.fixture
def user():
    return get_user_model().objects.create_user(username="alice")


@pytest.fixture
def other_user():
    return get_user_model().objects.create_user(username="bob")


# --- Model -----------------------------------------------------------------


@pytest.mark.django_db
def test_rating_is_linked_to_recipe_and_user(recipe, user):
    rating = RecipeRating.objects.create(recipe=recipe, user=user, stars=4)

    assert list(recipe.ratings.all()) == [rating]
    assert rating.user == user
    assert rating.created_at is not None


@pytest.mark.django_db
def test_user_can_rate_a_recipe_only_once(recipe, user):
    RecipeRating.objects.create(recipe=recipe, user=user, stars=4)

    with pytest.raises(IntegrityError):
        RecipeRating.objects.create(recipe=recipe, user=user, stars=5)


@pytest.mark.django_db
def test_different_users_can_rate_the_same_recipe(recipe, user, other_user):
    RecipeRating.objects.create(recipe=recipe, user=user, stars=4)
    RecipeRating.objects.create(recipe=recipe, user=other_user, stars=2)

    assert recipe.ratings.count() == 2


@pytest.mark.django_db
def test_ratings_are_ordered_newest_first(recipe, user, other_user):
    older = RecipeRating.objects.create(recipe=recipe, user=user, stars=3)
    newer = RecipeRating.objects.create(
        recipe=recipe, user=other_user, stars=5
    )
    # Pin timestamps so the test never depends on clock resolution.
    now = timezone.now()
    RecipeRating.objects.filter(pk=older.pk).update(
        created_at=now - timedelta(days=1)
    )
    RecipeRating.objects.filter(pk=newer.pk).update(created_at=now)

    assert list(RecipeRating.objects.all()) == [newer, older]


@pytest.mark.django_db
def test_ratings_are_deleted_with_their_recipe(recipe, user):
    RecipeRating.objects.create(recipe=recipe, user=user, stars=4)

    recipe.delete()

    assert RecipeRating.objects.count() == 0


# --- Serializer ------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize("stars", [1, 3, 5])
def test_serializer_accepts_stars_between_1_and_5(stars):
    serializer = RecipeRatingSerializer(data={"stars": stars})

    assert serializer.is_valid(), serializer.errors


@pytest.mark.django_db
@pytest.mark.parametrize("stars", [0, 6, -1])
def test_serializer_rejects_stars_outside_1_to_5(stars):
    serializer = RecipeRatingSerializer(data={"stars": stars})

    assert not serializer.is_valid()
    assert "stars" in serializer.errors


@pytest.mark.django_db
def test_serializer_saves_rating_with_recipe_and_user(recipe, user):
    serializer = RecipeRatingSerializer(data={"stars": 5})
    assert serializer.is_valid(), serializer.errors

    rating = serializer.save(recipe=recipe, user=user)

    assert rating.pk is not None
    assert (rating.recipe, rating.user, rating.stars) == (recipe, user, 5)


@pytest.mark.django_db
def test_serializer_ignores_read_only_fields(recipe, user):
    fixed_time = "2000-01-01T00:00:00Z"
    serializer = RecipeRatingSerializer(
        data={"id": 999, "stars": 4, "created_at": fixed_time}
    )
    assert serializer.is_valid(), serializer.errors

    rating = serializer.save(recipe=recipe, user=user)

    assert rating.pk != 999
    assert rating.created_at.year != 2000


@pytest.mark.django_db
def test_serializer_output_fields(recipe, user):
    rating = RecipeRating.objects.create(recipe=recipe, user=user, stars=2)

    data = RecipeRatingSerializer(rating).data

    assert set(data) == {"id", "stars", "created_at"}
    assert data["stars"] == 2
