from functools import update_wrapper
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from .helpers import _PackageBoundObject
from .templating import _default_template_ctx_processor
_sentinel = object()
def setupmethod(f):
    def wrapper_func(self, *args, **kwargs):
        if self._is_setup_finished():
            raise AssertionError(
                "A setup function was called after the "
                "first request was handled.  This usually indicates a bug "
                "in the application where a module was not imported "
                "and decorators or other functionality was called too late.\n"
                "To fix this make sure to import all your view modules, "
                "database models and everything related at a central place "
                "before the application starts serving requests."
            )
        return f(self, *args, **kwargs)
    return update_wrapper(wrapper_func, f)
class Scaffold(_PackageBoundObject):
    json_encoder = None
    json_decoder = None
    import_name = None
    template_folder = None
    root_path = None
    def __init__(
        self,
        import_name,
        static_folder="static",
        static_url_path=None,
        template_folder=None,
        root_path=None,
    ):
        super().__init__(
            import_name=import_name,
            template_folder=template_folder,
            root_path=root_path,
        )
        self.static_folder = static_folder
        self.static_url_path = static_url_path
        self.view_functions = {}
        self.error_handler_spec = {}
        self.before_request_funcs = {}
        self.after_request_funcs = {}
        self.teardown_request_funcs = {}
        self.template_context_processors = {None: [_default_template_ctx_processor]}
        self.url_value_preprocessors = {}
        self.url_default_functions = {}
    def _is_setup_finished(self):
        raise NotImplementedError
    def route(self, rule, **options):
        def decorator(f):
            endpoint = options.pop("endpoint", None)
            self.add_url_rule(rule, endpoint, f, **options)
            return f
        return decorator
    @setupmethod
    def add_url_rule(
        self,
        rule,
        endpoint=None,
        view_func=None,
        provide_automatic_options=None,
        **options,
    ):
        raise NotImplementedError
    def endpoint(self, endpoint):
        def decorator(f):
            self.view_functions[endpoint] = f
            return f
        return decorator
    @setupmethod
    def before_request(self, f):
        self.before_request_funcs.setdefault(None, []).append(f)
        return f
    @setupmethod
    def after_request(self, f):
        self.after_request_funcs.setdefault(None, []).append(f)
        return f
    @setupmethod
    def teardown_request(self, f):
        self.teardown_request_funcs.setdefault(None, []).append(f)
        return f
    @setupmethod
    def context_processor(self, f):
        self.template_context_processors[None].append(f)
        return f
    @setupmethod
    def url_value_preprocessor(self, f):
        self.url_value_preprocessors.setdefault(None, []).append(f)
        return f
    @setupmethod
    def url_defaults(self, f):
        self.url_default_functions.setdefault(None, []).append(f)
        return f
    @setupmethod
    def errorhandler(self, code_or_exception):
        def decorator(f):
            self._register_error_handler(None, code_or_exception, f)
            return f
        return decorator
    @setupmethod
    def register_error_handler(self, code_or_exception, f):
        self._register_error_handler(None, code_or_exception, f)
    @setupmethod
    def _register_error_handler(self, key, code_or_exception, f):
        if isinstance(code_or_exception, HTTPException):  
            raise ValueError(
                "Tried to register a handler for an exception instance"
                f" {code_or_exception!r}. Handlers can only be"
                " registered for exception classes or HTTP error codes."
            )
        try:
            exc_class, code = self._get_exc_class_and_code(code_or_exception)
        except KeyError:
            raise KeyError(
                f"'{code_or_exception}' is not a recognized HTTP error"
                " code. Use a subclass of HTTPException with that code"
                " instead."
            )
        handlers = self.error_handler_spec.setdefault(key, {}).setdefault(code, {})
        handlers[exc_class] = f
    @staticmethod
    def _get_exc_class_and_code(exc_class_or_code):
        if isinstance(exc_class_or_code, int):
            exc_class = default_exceptions[exc_class_or_code]
        else:
            exc_class = exc_class_or_code
        assert issubclass(
            exc_class, Exception
        ), "Custom exceptions must be subclasses of Exception."
        if issubclass(exc_class, HTTPException):
            return exc_class, exc_class.code
        else:
            return exc_class, None
def _endpoint_from_view_func(view_func):
    assert view_func is not None, "expected view func if endpoint is not provided."
    return view_func.__name__