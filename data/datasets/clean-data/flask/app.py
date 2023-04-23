import os
import sys
from datetime import timedelta
from itertools import chain
from threading import Lock
from werkzeug.datastructures import Headers
from werkzeug.datastructures import ImmutableDict
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import InternalServerError
from werkzeug.routing import BuildError
from werkzeug.routing import Map
from werkzeug.routing import RequestRedirect
from werkzeug.routing import RoutingException
from werkzeug.routing import Rule
from werkzeug.wrappers import BaseResponse
from . import cli
from . import json
from .config import Config
from .config import ConfigAttribute
from .ctx import _AppCtxGlobals
from .ctx import AppContext
from .ctx import RequestContext
from .globals import _request_ctx_stack
from .globals import g
from .globals import request
from .globals import session
from .helpers import find_package
from .helpers import get_debug_flag
from .helpers import get_env
from .helpers import get_flashed_messages
from .helpers import get_load_dotenv
from .helpers import locked_cached_property
from .helpers import url_for
from .json import jsonify
from .logging import create_logger
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .scaffold import setupmethod
from .sessions import SecureCookieSessionInterface
from .signals import appcontext_tearing_down
from .signals import got_request_exception
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .templating import DispatchingJinjaLoader
from .templating import Environment
from .wrappers import Request
from .wrappers import Response
def _make_timedelta(value):
    if not isinstance(value, timedelta):
        return timedelta(seconds=value)
    return value
class Flask(Scaffold):
    request_class = Request
    response_class = Response
    jinja_environment = Environment
    app_ctx_globals_class = _AppCtxGlobals
    config_class = Config
    testing = ConfigAttribute("TESTING")
    secret_key = ConfigAttribute("SECRET_KEY")
    session_cookie_name = ConfigAttribute("SESSION_COOKIE_NAME")
    permanent_session_lifetime = ConfigAttribute(
        "PERMANENT_SESSION_LIFETIME", get_converter=_make_timedelta
    )
    send_file_max_age_default = ConfigAttribute(
        "SEND_FILE_MAX_AGE_DEFAULT", get_converter=_make_timedelta
    )
    use_x_sendfile = ConfigAttribute("USE_X_SENDFILE")
    json_encoder = json.JSONEncoder
    json_decoder = json.JSONDecoder
    jinja_options = {"extensions": ["jinja2.ext.autoescape", "jinja2.ext.with_"]}
    default_config = ImmutableDict(
        {
            "ENV": None,
            "DEBUG": None,
            "TESTING": False,
            "PROPAGATE_EXCEPTIONS": None,
            "PRESERVE_CONTEXT_ON_EXCEPTION": None,
            "SECRET_KEY": None,
            "PERMANENT_SESSION_LIFETIME": timedelta(days=31),
            "USE_X_SENDFILE": False,
            "SERVER_NAME": None,
            "APPLICATION_ROOT": "/",
            "SESSION_COOKIE_NAME": "session",
            "SESSION_COOKIE_DOMAIN": None,
            "SESSION_COOKIE_PATH": None,
            "SESSION_COOKIE_HTTPONLY": True,
            "SESSION_COOKIE_SECURE": False,
            "SESSION_COOKIE_SAMESITE": None,
            "SESSION_REFRESH_EACH_REQUEST": True,
            "MAX_CONTENT_LENGTH": None,
            "SEND_FILE_MAX_AGE_DEFAULT": timedelta(hours=12),
            "TRAP_BAD_REQUEST_ERRORS": None,
            "TRAP_HTTP_EXCEPTIONS": False,
            "EXPLAIN_TEMPLATE_LOADING": False,
            "PREFERRED_URL_SCHEME": "http",
            "JSON_AS_ASCII": True,
            "JSON_SORT_KEYS": True,
            "JSONIFY_PRETTYPRINT_REGULAR": False,
            "JSONIFY_MIMETYPE": "application/json",
            "TEMPLATES_AUTO_RELOAD": None,
            "MAX_COOKIE_SIZE": 4093,
        }
    )
    url_rule_class = Rule
    url_map_class = Map
    test_client_class = None
    test_cli_runner_class = None
    session_interface = SecureCookieSessionInterface()
    import_name = None
    template_folder = None
    root_path = None
    def __init__(
        self,
        import_name,
        static_url_path=None,
        static_folder="static",
        static_host=None,
        host_matching=False,
        subdomain_matching=False,
        template_folder="templates",
        instance_path=None,
        instance_relative_config=False,
        root_path=None,
    ):
        super().__init__(
            import_name=import_name,
            static_folder=static_folder,
            static_url_path=static_url_path,
            template_folder=template_folder,
            root_path=root_path,
        )
        if instance_path is None:
            instance_path = self.auto_find_instance_path()
        elif not os.path.isabs(instance_path):
            raise ValueError(
                "If an instance path is provided it must be absolute."
                " A relative path was given instead."
            )
        self.instance_path = instance_path
        self.config = self.make_config(instance_relative_config)
        self.url_build_error_handlers = []
        self.before_first_request_funcs = []
        self.teardown_appcontext_funcs = []
        self.shell_context_processors = []
        self.blueprints = {}
        self._blueprint_order = []
        self.extensions = {}
        self.url_map = self.url_map_class()
        self.url_map.host_matching = host_matching
        self.subdomain_matching = subdomain_matching
        self._got_first_request = False
        self._before_request_lock = Lock()
        if self.has_static_folder:
            assert (
                bool(static_host) == host_matching
            ), "Invalid static_host/host_matching combination"
            self.add_url_rule(
                f"{self.static_url_path}/<path:filename>",
                endpoint="static",
                host=static_host,
                view_func=self.send_static_file,
            )
        self.cli.name = self.name
    def _is_setup_finished(self):
        return self.debug and self._got_first_request
    @locked_cached_property
    def name(self):
        if self.import_name == "__main__":
            fn = getattr(sys.modules["__main__"], "__file__", None)
            if fn is None:
                return "__main__"
            return os.path.splitext(os.path.basename(fn))[0]
        return self.import_name
    @property
    def propagate_exceptions(self):
        rv = self.config["PROPAGATE_EXCEPTIONS"]
        if rv is not None:
            return rv
        return self.testing or self.debug
    @property
    def preserve_context_on_exception(self):
        rv = self.config["PRESERVE_CONTEXT_ON_EXCEPTION"]
        if rv is not None:
            return rv
        return self.debug
    @locked_cached_property
    def logger(self):
        return create_logger(self)
    @locked_cached_property
    def jinja_env(self):
        return self.create_jinja_environment()
    @property
    def got_first_request(self):
        return self._got_first_request
    def make_config(self, instance_relative=False):
        root_path = self.root_path
        if instance_relative:
            root_path = self.instance_path
        defaults = dict(self.default_config)
        defaults["ENV"] = get_env()
        defaults["DEBUG"] = get_debug_flag()
        return self.config_class(root_path, defaults)
    def auto_find_instance_path(self):
        prefix, package_path = find_package(self.import_name)
        if prefix is None:
            return os.path.join(package_path, "instance")
        return os.path.join(prefix, "var", f"{self.name}-instance")
    def open_instance_resource(self, resource, mode="rb"):
        return open(os.path.join(self.instance_path, resource), mode)
    @property
    def templates_auto_reload(self):
        rv = self.config["TEMPLATES_AUTO_RELOAD"]
        return rv if rv is not None else self.debug
    @templates_auto_reload.setter
    def templates_auto_reload(self, value):
        self.config["TEMPLATES_AUTO_RELOAD"] = value
    def create_jinja_environment(self):
        options = dict(self.jinja_options)
        if "autoescape" not in options:
            options["autoescape"] = self.select_jinja_autoescape
        if "auto_reload" not in options:
            options["auto_reload"] = self.templates_auto_reload
        rv = self.jinja_environment(self, **options)
        rv.globals.update(
            url_for=url_for,
            get_flashed_messages=get_flashed_messages,
            config=self.config,
            request=request,
            session=session,
            g=g,
        )
        rv.filters["tojson"] = json.tojson_filter
        return rv
    def create_global_jinja_loader(self):
        return DispatchingJinjaLoader(self)
    def select_jinja_autoescape(self, filename):
        if filename is None:
            return True
        return filename.endswith((".html", ".htm", ".xml", ".xhtml"))
    def update_template_context(self, context):
        funcs = self.template_context_processors[None]
        reqctx = _request_ctx_stack.top
        if reqctx is not None:
            bp = reqctx.request.blueprint
            if bp is not None and bp in self.template_context_processors:
                funcs = chain(funcs, self.template_context_processors[bp])
        orig_ctx = context.copy()
        for func in funcs:
            context.update(func())
        context.update(orig_ctx)
    def make_shell_context(self):
        rv = {"app": self, "g": g}
        for processor in self.shell_context_processors:
            rv.update(processor())
        return rv
    env = ConfigAttribute("ENV")
    @property
    def debug(self):
        return self.config["DEBUG"]
    @debug.setter
    def debug(self, value):
        self.config["DEBUG"] = value
        self.jinja_env.auto_reload = self.templates_auto_reload
    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        if os.environ.get("FLASK_RUN_FROM_CLI") == "true":
            from .debughelpers import explain_ignored_app_run
            explain_ignored_app_run()
            return
        if get_load_dotenv(load_dotenv):
            cli.load_dotenv()
            if "FLASK_ENV" in os.environ:
                self.env = get_env()
                self.debug = get_debug_flag()
            elif "FLASK_DEBUG" in os.environ:
                self.debug = get_debug_flag()
        if debug is not None:
            self.debug = bool(debug)
        server_name = self.config.get("SERVER_NAME")
        sn_host = sn_port = None
        if server_name:
            sn_host, _, sn_port = server_name.partition(":")
        if not host:
            if sn_host:
                host = sn_host
            else:
                host = "127.0.0.1"
        if port or port == 0:
            port = int(port)
        elif sn_port:
            port = int(sn_port)
        else:
            port = 5000
        options.setdefault("use_reloader", self.debug)
        options.setdefault("use_debugger", self.debug)
        options.setdefault("threaded", True)
        cli.show_server_banner(self.env, self.debug, self.name, False)
        from werkzeug.serving import run_simple
        try:
            run_simple(host, port, self, **options)
        finally:
            self._got_first_request = False
    def test_client(self, use_cookies=True, **kwargs):
        cls = self.test_client_class
        if cls is None:
            from .testing import FlaskClient as cls
        return cls(self, self.response_class, use_cookies=use_cookies, **kwargs)
    def test_cli_runner(self, **kwargs):
        cls = self.test_cli_runner_class
        if cls is None:
            from .testing import FlaskCliRunner as cls
        return cls(self, **kwargs)
    @setupmethod
    def register_blueprint(self, blueprint, **options):
        first_registration = False
        if blueprint.name in self.blueprints:
            assert self.blueprints[blueprint.name] is blueprint, (
                "A name collision occurred between blueprints"
                f" {blueprint!r} and {self.blueprints[blueprint.name]!r}."
                f" Both share the same name {blueprint.name!r}."
                f" Blueprints that are created on the fly need unique"
                f" names."
            )
        else:
            self.blueprints[blueprint.name] = blueprint
            self._blueprint_order.append(blueprint)
            first_registration = True
        blueprint.register(self, options, first_registration)
    def iter_blueprints(self):
        return iter(self._blueprint_order)
    @setupmethod
    def add_url_rule(
        self,
        rule,
        endpoint=None,
        view_func=None,
        provide_automatic_options=None,
        **options,
    ):
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)
        options["endpoint"] = endpoint
        methods = options.pop("methods", None)
        if methods is None:
            methods = getattr(view_func, "methods", None) or ("GET",)
        if isinstance(methods, str):
            raise TypeError(
                "Allowed methods must be a list of strings, for"
                ' example: @app.route(..., methods=["POST"])'
            )
        methods = {item.upper() for item in methods}
        required_methods = set(getattr(view_func, "required_methods", ()))
        if provide_automatic_options is None:
            provide_automatic_options = getattr(
                view_func, "provide_automatic_options", None
            )
        if provide_automatic_options is None:
            if "OPTIONS" not in methods:
                provide_automatic_options = True
                required_methods.add("OPTIONS")
            else:
                provide_automatic_options = False
        methods |= required_methods
        rule = self.url_rule_class(rule, methods=methods, **options)
        rule.provide_automatic_options = provide_automatic_options
        self.url_map.add(rule)
        if view_func is not None:
            old_func = self.view_functions.get(endpoint)
            if old_func is not None and old_func != view_func:
                raise AssertionError(
                    "View function mapping is overwriting an existing"
                    f" endpoint function: {endpoint}"
                )
            self.view_functions[endpoint] = view_func
    @setupmethod
    def template_filter(self, name=None):
        def decorator(f):
            self.add_template_filter(f, name=name)
            return f
        return decorator
    @setupmethod
    def add_template_filter(self, f, name=None):
        self.jinja_env.filters[name or f.__name__] = f
    @setupmethod
    def template_test(self, name=None):
        def decorator(f):
            self.add_template_test(f, name=name)
            return f
        return decorator
    @setupmethod
    def add_template_test(self, f, name=None):
        self.jinja_env.tests[name or f.__name__] = f
    @setupmethod
    def template_global(self, name=None):
        def decorator(f):
            self.add_template_global(f, name=name)
            return f
        return decorator
    @setupmethod
    def add_template_global(self, f, name=None):
        self.jinja_env.globals[name or f.__name__] = f
    @setupmethod
    def before_first_request(self, f):
        self.before_first_request_funcs.append(f)
        return f
    @setupmethod
    def teardown_appcontext(self, f):
        self.teardown_appcontext_funcs.append(f)
        return f
    @setupmethod
    def shell_context_processor(self, f):
        self.shell_context_processors.append(f)
        return f
    def _find_error_handler(self, e):
        exc_class, code = self._get_exc_class_and_code(type(e))
        for name, c in (
            (request.blueprint, code),
            (None, code),
            (request.blueprint, None),
            (None, None),
        ):
            handler_map = self.error_handler_spec.setdefault(name, {}).get(c)
            if not handler_map:
                continue
            for cls in exc_class.__mro__:
                handler = handler_map.get(cls)
                if handler is not None:
                    return handler
    def handle_http_exception(self, e):
        if e.code is None:
            return e
        if isinstance(e, RoutingException):
            return e
        handler = self._find_error_handler(e)
        if handler is None:
            return e
        return handler(e)
    def trap_http_exception(self, e):
        if self.config["TRAP_HTTP_EXCEPTIONS"]:
            return True
        trap_bad_request = self.config["TRAP_BAD_REQUEST_ERRORS"]
        if (
            trap_bad_request is None
            and self.debug
            and isinstance(e, BadRequestKeyError)
        ):
            return True
        if trap_bad_request:
            return isinstance(e, BadRequest)
        return False
    def handle_user_exception(self, e):
        if isinstance(e, BadRequestKeyError):
            if self.debug or self.config["TRAP_BAD_REQUEST_ERRORS"]:
                e.show_exception = True
                if e.args[0] not in e.get_description():
                    e.description = f"KeyError: {e.args[0]!r}"
            elif not hasattr(BadRequestKeyError, "show_exception"):
                e.args = ()
        if isinstance(e, HTTPException) and not self.trap_http_exception(e):
            return self.handle_http_exception(e)
        handler = self._find_error_handler(e)
        if handler is None:
            raise
        return handler(e)
    def handle_exception(self, e):
        exc_info = sys.exc_info()
        got_request_exception.send(self, exception=e)
        if self.propagate_exceptions:
            if exc_info[1] is e:
                raise
            raise e
        self.log_exception(exc_info)
        server_error = InternalServerError()
        server_error.original_exception = e
        handler = self._find_error_handler(server_error)
        if handler is not None:
            server_error = handler(server_error)
        return self.finalize_request(server_error, from_error_handler=True)
    def log_exception(self, exc_info):
        self.logger.error(
            f"Exception on {request.path} [{request.method}]", exc_info=exc_info
        )
    def raise_routing_exception(self, request):
        if (
            not self.debug
            or not isinstance(request.routing_exception, RequestRedirect)
            or request.method in ("GET", "HEAD", "OPTIONS")
        ):
            raise request.routing_exception
        from .debughelpers import FormDataRoutingRedirect
        raise FormDataRoutingRedirect(request)
    def dispatch_request(self):
        req = _request_ctx_stack.top.request
        if req.routing_exception is not None:
            self.raise_routing_exception(req)
        rule = req.url_rule
        if (
            getattr(rule, "provide_automatic_options", False)
            and req.method == "OPTIONS"
        ):
            return self.make_default_options_response()
        return self.view_functions[rule.endpoint](**req.view_args)
    def full_dispatch_request(self):
        self.try_trigger_before_first_request_functions()
        try:
            request_started.send(self)
            rv = self.preprocess_request()
            if rv is None:
                rv = self.dispatch_request()
        except Exception as e:
            rv = self.handle_user_exception(e)
        return self.finalize_request(rv)
    def finalize_request(self, rv, from_error_handler=False):
        response = self.make_response(rv)
        try:
            response = self.process_response(response)
            request_finished.send(self, response=response)
        except Exception:
            if not from_error_handler:
                raise
            self.logger.exception(
                "Request finalizing failed with an error while handling an error"
            )
        return response
    def try_trigger_before_first_request_functions(self):
        if self._got_first_request:
            return
        with self._before_request_lock:
            if self._got_first_request:
                return
            for func in self.before_first_request_funcs:
                func()
            self._got_first_request = True
    def make_default_options_response(self):
        adapter = _request_ctx_stack.top.url_adapter
        methods = adapter.allowed_methods()
        rv = self.response_class()
        rv.allow.update(methods)
        return rv
    def should_ignore_error(self, error):
        return False
    def make_response(self, rv):
        status = headers = None
        if isinstance(rv, tuple):
            len_rv = len(rv)
            if len_rv == 3:
                rv, status, headers = rv
            elif len_rv == 2:
                if isinstance(rv[1], (Headers, dict, tuple, list)):
                    rv, headers = rv
                else:
                    rv, status = rv
            else:
                raise TypeError(
                    "The view function did not return a valid response tuple."
                    " The tuple must have the form (body, status, headers),"
                    " (body, status), or (body, headers)."
                )
        if rv is None:
            raise TypeError(
                f"The view function for {request.endpoint!r} did not"
                " return a valid response. The function either returned"
                " None or ended without a return statement."
            )
        if not isinstance(rv, self.response_class):
            if isinstance(rv, (str, bytes, bytearray)):
                rv = self.response_class(rv, status=status, headers=headers)
                status = headers = None
            elif isinstance(rv, dict):
                rv = jsonify(rv)
            elif isinstance(rv, BaseResponse) or callable(rv):
                try:
                    rv = self.response_class.force_type(rv, request.environ)
                except TypeError as e:
                    raise TypeError(
                        f"{e}\nThe view function did not return a valid"
                        " response. The return type must be a string,"
                        " dict, tuple, Response instance, or WSGI"
                        f" callable, but it was a {type(rv).__name__}."
                    ).with_traceback(sys.exc_info()[2])
            else:
                raise TypeError(
                    "The view function did not return a valid"
                    " response. The return type must be a string,"
                    " dict, tuple, Response instance, or WSGI"
                    f" callable, but it was a {type(rv).__name__}."
                )
        if status is not None:
            if isinstance(status, (str, bytes, bytearray)):
                rv.status = status
            else:
                rv.status_code = status
        if headers:
            rv.headers.update(headers)
        return rv
    def create_url_adapter(self, request):
        if request is not None:
            if not self.subdomain_matching:
                subdomain = self.url_map.default_subdomain or None
            else:
                subdomain = None
            return self.url_map.bind_to_environ(
                request.environ,
                server_name=self.config["SERVER_NAME"],
                subdomain=subdomain,
            )
        if self.config["SERVER_NAME"] is not None:
            return self.url_map.bind(
                self.config["SERVER_NAME"],
                script_name=self.config["APPLICATION_ROOT"],
                url_scheme=self.config["PREFERRED_URL_SCHEME"],
            )
    def inject_url_defaults(self, endpoint, values):
        funcs = self.url_default_functions.get(None, ())
        if "." in endpoint:
            bp = endpoint.rsplit(".", 1)[0]
            funcs = chain(funcs, self.url_default_functions.get(bp, ()))
        for func in funcs:
            func(endpoint, values)
    def handle_url_build_error(self, error, endpoint, values):
        for handler in self.url_build_error_handlers:
            try:
                rv = handler(error, endpoint, values)
            except BuildError as e:
                error = e
            else:
                if rv is not None:
                    return rv
        if error is sys.exc_info()[1]:
            raise
        raise error
    def preprocess_request(self):
        bp = _request_ctx_stack.top.request.blueprint
        funcs = self.url_value_preprocessors.get(None, ())
        if bp is not None and bp in self.url_value_preprocessors:
            funcs = chain(funcs, self.url_value_preprocessors[bp])
        for func in funcs:
            func(request.endpoint, request.view_args)
        funcs = self.before_request_funcs.get(None, ())
        if bp is not None and bp in self.before_request_funcs:
            funcs = chain(funcs, self.before_request_funcs[bp])
        for func in funcs:
            rv = func()
            if rv is not None:
                return rv
    def process_response(self, response):
        ctx = _request_ctx_stack.top
        bp = ctx.request.blueprint
        funcs = ctx._after_request_functions
        if bp is not None and bp in self.after_request_funcs:
            funcs = chain(funcs, reversed(self.after_request_funcs[bp]))
        if None in self.after_request_funcs:
            funcs = chain(funcs, reversed(self.after_request_funcs[None]))
        for handler in funcs:
            response = handler(response)
        if not self.session_interface.is_null_session(ctx.session):
            self.session_interface.save_session(self, ctx.session, response)
        return response
    def do_teardown_request(self, exc=_sentinel):
        if exc is _sentinel:
            exc = sys.exc_info()[1]
        funcs = reversed(self.teardown_request_funcs.get(None, ()))
        bp = _request_ctx_stack.top.request.blueprint
        if bp is not None and bp in self.teardown_request_funcs:
            funcs = chain(funcs, reversed(self.teardown_request_funcs[bp]))
        for func in funcs:
            func(exc)
        request_tearing_down.send(self, exc=exc)
    def do_teardown_appcontext(self, exc=_sentinel):
        if exc is _sentinel:
            exc = sys.exc_info()[1]
        for func in reversed(self.teardown_appcontext_funcs):
            func(exc)
        appcontext_tearing_down.send(self, exc=exc)
    def app_context(self):
        return AppContext(self)
    def request_context(self, environ):
        return RequestContext(self, environ)
    def test_request_context(self, *args, **kwargs):
        from .testing import EnvironBuilder
        builder = EnvironBuilder(self, *args, **kwargs)
        try:
            return self.request_context(builder.get_environ())
        finally:
            builder.close()
    def wsgi_app(self, environ, start_response):
        ctx = self.request_context(environ)
        error = None
        try:
            try:
                ctx.push()
                response = self.full_dispatch_request()
            except Exception as e:
                error = e
                response = self.handle_exception(e)
            except:  
                error = sys.exc_info()[1]
                raise
            return response(environ, start_response)
        finally:
            if self.should_ignore_error(error):
                error = None
            ctx.auto_pop(error)
    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)
    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r}>"