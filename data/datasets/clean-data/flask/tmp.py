from datetime import timedelta
from werkzeug.datastructures import ImmutableDict
from werkzeug.routing import Map
from werkzeug.routing import Rule
from . import json
from .config import Config
from .config import ConfigAttribute
from .ctx import _AppCtxGlobals
from .scaffold import Scaffold
from .sessions import SecureCookieSessionInterface
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
            reveal_type(self)