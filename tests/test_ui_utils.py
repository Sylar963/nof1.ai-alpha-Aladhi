"""Tests for small NiceGUI utility helpers that don't need a live client."""

from types import SimpleNamespace

from src.gui.services.ui_utils import safe_notify


class _FakeOutbox:
    def __init__(self):
        self.calls = []

    def enqueue_message(self, message_type, data, target_id):
        self.calls.append((message_type, data, target_id))


def test_safe_notify_enqueues_for_live_client():
    outbox = _FakeOutbox()
    client = SimpleNamespace(
        _deleted=False,
        has_socket_connection=True,
        id="client-1",
        outbox=outbox,
    )

    sent = safe_notify(client, "hello", type="positive", close_button=True, multi_line=True)

    assert sent is True
    assert outbox.calls == [
        (
            "notify",
            {
                "message": "hello",
                "type": "positive",
                "closeBtn": True,
                "multiLine": True,
            },
            "client-1",
        )
    ]


def test_safe_notify_skips_deleted_or_disconnected_clients():
    deleted_client = SimpleNamespace(
        _deleted=True,
        has_socket_connection=True,
        id="client-1",
        outbox=_FakeOutbox(),
    )
    disconnected_client = SimpleNamespace(
        _deleted=False,
        has_socket_connection=False,
        id="client-2",
        outbox=_FakeOutbox(),
    )

    assert safe_notify(deleted_client, "x") is False
    assert safe_notify(disconnected_client, "x") is False
