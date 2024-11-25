def test_create_object_from_config():
    from frogbox.config import ObjectDefinition, create_object_from_config

    obj_def = ObjectDefinition(
        class_name="datetime.timedelta",
        params={
            "hours": 2,
            "minutes": 34,
        }
    )
    obj = create_object_from_config(obj_def)
    assert obj.seconds == 2 * 60**2 + 34 * 60
