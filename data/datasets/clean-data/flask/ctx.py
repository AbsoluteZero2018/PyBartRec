import sys
from functools import update_wrapper
from werkzeug.exceptions import HTTPException
from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .signals import appcontext_popped
from .signals import appcontext_pushed
_sentinel = object()
class _AppCtxGlobals:
    def get(self, name, default=None):
        return self.__dict__.get(name, default)
    def pop(self, name, default=_sentinel):
        if default is _sentinel:
            return self.__dict__.pop(name)
        else:
            return self.__dict__.pop(name, default)
    def setdefault(self, name, default=None):
        return self.__dict__.setdefault(name, default)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)
    def __repr__(self):
        top = _app_ctx_stack.top
        if top is not None:
            return f"<flask.g of {top.app.name!r}>"
        return object.__repr__(self)
def after_this_request(f):
    _request_ctx_stack.top._after_request_functions.append(f)
    return f
def copy_current_request_context(f):
    top = _request_ctx_stack.top
    if top is None:
        raise RuntimeError(
            "This decorator can only be used at local scopes "
            "when a request context is on the stack.  For instance within "
            "view functions."
        )
    reqctx = top.copy()
    def wrapper(*args, **kwargs):
        with reqctx:
            return f(*args, **kwargs)
    return update_wrapper(wrapper, f)
def has_request_context():
    return _request_ctx_stack.top is not None
def has_app_context():
    return _app_ctx_stack.top is not None
class AppContext:
    def __init__(self, app):
        self.app = app
        self.url_adapter = app.create_url_adapter(None)
        self.g = app.app_ctx_globals_class()
        self._refcnt = 0
    def push(self):
        self._refcnt += 1
        _app_ctx_stack.push(self)
        appcontext_pushed.send(self.app)
    def pop(self, exc=_sentinel):
        try:
            self._refcnt -= 1
            if self._refcnt <= 0:
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_appcontext(exc)
        finally:
            rv = _app_ctx_stack.pop()
        assert rv is self, f"Popped wrong app context.  ({rv!r} instead of {self!r})"
        appcontext_popped.send(self.app)
    def __enter__(self):
        self.push()
        return self
    def __exit__(self, exc_type, exc_value, tb):
        self.pop(exc_value)
class RequestContext:
    def __init__(self, app, environ, request=None, session=None):
        self.app = app
        if request is None:
            request = app.request_class(environ)
        self.request = request
        self.url_adapter = None
        try:
            self.url_adapter = app.create_url_adapter(self.request)
        except HTTPException as e:
            self.request.routing_exception = e
        self.flashes = None
        self.session = session
        self._implicit_app_ctx_stack = []
        self.preserved = False
        self._preserved_exc = None
        self._after_request_functions = []
    @property
    def g(self):
        return _app_ctx_stack.top.g
    @g.setter
    def g(self, value):
        _app_ctx_stack.top.g = value
    def copy(self):
        return self.__class__(
            self.app,
            environ=self.request.environ,
            request=self.request,
            session=self.session,
        )
    def match_request(self):
        try:
            result = self.url_adapter.match(return_rule=True)
            self.request.url_rule, self.request.view_args = result
        except HTTPException as e:
            self.request.routing_exception = e
    def push(self):
        top = _request_ctx_stack.top
        if top is not None and top.preserved:
            top.pop(top._preserved_exc)
        app_ctx = _app_ctx_stack.top
        if app_ctx is None or app_ctx.app != self.app:
            app_ctx = self.app.app_context()
            app_ctx.push()
            self._implicit_app_ctx_stack.append(app_ctx)
        else:
            self._implicit_app_ctx_stack.append(None)
        _request_ctx_stack.push(self)
        if self.session is None:
            session_interface = self.app.session_interface
            self.session = session_interface.open_session(self.app, self.request)
            if self.session is None:
                self.session = session_interface.make_null_session(self.app)
        if self.url_adapter is not None:
            self.match_request()
    def pop(self, exc=_sentinel):
        app_ctx = self._implicit_app_ctx_stack.pop()
        clear_request = False
        try:
            if not self._implicit_app_ctx_stack:
                self.preserved = False
                self._preserved_exc = None
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_request(exc)
                request_close = getattr(self.request, "close", None)
                if request_close is not None:
                    request_close()
                clear_request = True
        finally:
            rv = _request_ctx_stack.pop()
            if clear_request:
                rv.request.environ["werkzeug.request"] = None
            if app_ctx is not None:
                app_ctx.pop(exc)
            assert (
                rv is self
            ), f"Popped wrong request context. ({rv!r} instead of {self!r})"
    def auto_pop(self, exc):
        if self.request.environ.get("flask._preserve_context") or (
            exc is not None and self.app.preserve_context_on_exception
        ):
            self.preserved = True
            self._preserved_exc = exc
        else:
            self.pop(exc)
    def __enter__(self):
        self.push()
        return self
    def __exit__(self, exc_type, exc_value, tb):
        self.auto_pop(exc_value)
    def __repr__(self):
        return (
            f"<{type(self).__name__} {self.request.url!r}"
            f" [{self.request.method}] of {self.app.name}>"
        )