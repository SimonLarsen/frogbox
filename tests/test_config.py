from frogbox.config import ObjectDefinition, create_object_from_config


def test_create_object_from_config():
    obj_def = ObjectDefinition(
        object="datetime.timedelta",
        kwargs={
            "hours": 2,
            "minutes": 34,
        }
    )
    obj = create_object_from_config(obj_def)
    assert obj.seconds == 2 * 60**2 + 34 * 60


def test_create_function_from_config():
    fun_def = ObjectDefinition(
        function="builtins.sorted",
        kwargs={"reverse": True},
    )
    fun = create_object_from_config(fun_def)
    assert fun([4, 2, 3, 5, 1]) == [5, 4, 3, 2, 1]


def test_create_lambda_from_config():
    lambda_def = ObjectDefinition(
        lambda_="x, y, z: x + y + z",
        args=[7],
    )
    fun = create_object_from_config(lambda_def)
    assert fun(9, 13) == 7 + 9 + 13
