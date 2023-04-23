try:
    from blinker import Namespace
    signals_available = True
except ImportError:
    signals_available = False
    class Namespace:
        def signal(self, name, doc=None):
            return _FakeSignal(name, doc)
    class _FakeSignal:
        def __init__(self, name, doc=None):
            self.name = name
            self.__doc__ = doc
        def send(self, *args, **kwargs):
            pass
        def _fail(self, *args, **kwargs):
            raise RuntimeError(
                "Signalling support is unavailable because the blinker"
                " library is not installed."
            )
        connect = connect_via = connected_to = temporarily_connected_to = _fail
        disconnect = _fail
        has_receivers_for = receivers_for = _fail
        del _fail
_signals = Namespace()
template_rendered = _signals.signal("template-rendered")
before_render_template = _signals.signal("before-render-template")
request_started = _signals.signal("request-started")
request_finished = _signals.signal("request-finished")
request_tearing_down = _signals.signal("request-tearing-down")
got_request_exception = _signals.signal("got-request-exception")
appcontext_tearing_down = _signals.signal("appcontext-tearing-down")
appcontext_pushed = _signals.signal("appcontext-pushed")
appcontext_popped = _signals.signal("appcontext-popped")
message_flashed = _signals.signal("message-flashed")