from uniscan.ui.camera_health import camera_health_state


def test_camera_health_closed() -> None:
    state = camera_health_state(is_open=False, is_previewing=False)
    assert state.label == "Camera: Closed"


def test_camera_health_open() -> None:
    state = camera_health_state(is_open=True, is_previewing=False)
    assert state.label == "Camera: Open"


def test_camera_health_previewing() -> None:
    state = camera_health_state(is_open=True, is_previewing=True)
    assert state.label == "Camera: Previewing"


def test_camera_health_error_overrides_other_states() -> None:
    state = camera_health_state(is_open=True, is_previewing=True, error_text="fail")
    assert state.label == "Camera: Error"
