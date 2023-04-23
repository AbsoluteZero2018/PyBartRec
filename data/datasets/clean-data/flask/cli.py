import ast
import inspect
import os
import platform
import re
import sys
import traceback
import warnings
from functools import update_wrapper
from operator import attrgetter
from threading import Lock
from threading import Thread
import click
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_env
from .helpers import get_load_dotenv
try:
    import dotenv
except ImportError:
    dotenv = None
try:
    import ssl
except ImportError:
    ssl = None
class NoAppException(click.UsageError):
def find_best_app(script_info, module):
    from . import Flask
    for attr_name in ("app", "application"):
        app = getattr(module, attr_name, None)
        if isinstance(app, Flask):
            return app
    matches = [v for v in module.__dict__.values() if isinstance(v, Flask)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise NoAppException(
            "Detected multiple Flask applications in module"
            f" {module.__name__!r}. Use 'FLASK_APP={module.__name__}:name'"
            f" to specify the correct one."
        )
    for attr_name in {"create_app", "make_app"}:
        app_factory = getattr(module, attr_name, None)
        if inspect.isfunction(app_factory):
            try:
                app = call_factory(script_info, app_factory)
                if isinstance(app, Flask):
                    return app
            except TypeError:
                if not _called_with_wrong_args(app_factory):
                    raise
                raise NoAppException(
                    f"Detected factory {attr_name!r} in module {module.__name__!r},"
                    " but could not call it without arguments. Use"
                    f" \"FLASK_APP='{module.__name__}:{attr_name}(args)'\""
                    " to specify arguments."
                )
    raise NoAppException(
        "Failed to find Flask application or factory in module"
        f" {module.__name__!r}. Use 'FLASK_APP={module.__name__}:name'"
        " to specify one."
    )
def call_factory(script_info, app_factory, args=None, kwargs=None):
    sig = inspect.signature(app_factory)
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    if "script_info" in sig.parameters:
        warnings.warn(
            "The 'script_info' argument is deprecated and will not be"
            " passed to the app factory function in 2.1.",
            DeprecationWarning,
        )
        kwargs["script_info"] = script_info
    if (
        not args
        and len(sig.parameters) == 1
        and next(iter(sig.parameters.values())).default is inspect.Parameter.empty
    ):
        warnings.warn(
            "Script info is deprecated and will not be passed as the"
            " single argument to the app factory function in 2.1.",
            DeprecationWarning,
        )
        args.append(script_info)
    return app_factory(*args, **kwargs)
def _called_with_wrong_args(f):
    tb = sys.exc_info()[2]
    try:
        while tb is not None:
            if tb.tb_frame.f_code is f.__code__:
                return False
            tb = tb.tb_next
        return True
    finally:
        del tb
def find_app_by_string(script_info, module, app_name):
    from . import Flask
    try:
        expr = ast.parse(app_name.strip(), mode="eval").body
    except SyntaxError:
        raise NoAppException(
            f"Failed to parse {app_name!r} as an attribute name or function call."
        )
    if isinstance(expr, ast.Name):
        name = expr.id
        args = kwargs = None
    elif isinstance(expr, ast.Call):
        if not isinstance(expr.func, ast.Name):
            raise NoAppException(
                f"Function reference must be a simple name: {app_name!r}."
            )
        name = expr.func.id
        try:
            args = [ast.literal_eval(arg) for arg in expr.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords}
        except ValueError:
            raise NoAppException(
                f"Failed to parse arguments as literal values: {app_name!r}."
            )
    else:
        raise NoAppException(
            f"Failed to parse {app_name!r} as an attribute name or function call."
        )
    try:
        attr = getattr(module, name)
    except AttributeError:
        raise NoAppException(
            f"Failed to find attribute {name!r} in {module.__name__!r}."
        )
    if inspect.isfunction(attr):
        try:
            app = call_factory(script_info, attr, args, kwargs)
        except TypeError:
            if not _called_with_wrong_args(attr):
                raise
            raise NoAppException(
                f"The factory {app_name!r} in module"
                f" {module.__name__!r} could not be called with the"
                " specified arguments."
            )
    else:
        app = attr
    if isinstance(app, Flask):
        return app
    raise NoAppException(
        "A valid Flask application was not obtained from"
        f" '{module.__name__}:{app_name}'."
    )
def prepare_import(path):
    path = os.path.realpath(path)
    fname, ext = os.path.splitext(path)
    if ext == ".py":
        path = fname
    if os.path.basename(path) == "__init__":
        path = os.path.dirname(path)
    module_name = []
    while True:
        path, name = os.path.split(path)
        module_name.append(name)
        if not os.path.exists(os.path.join(path, "__init__.py")):
            break
    if sys.path[0] != path:
        sys.path.insert(0, path)
    return ".".join(module_name[::-1])
def locate_app(script_info, module_name, app_name, raise_if_not_found=True):
    __traceback_hide__ = True  
    try:
        __import__(module_name)
    except ImportError:
        if sys.exc_info()[2].tb_next:
            raise NoAppException(
                f"While importing {module_name!r}, an ImportError was"
                f" raised:\n\n{traceback.format_exc()}"
            )
        elif raise_if_not_found:
            raise NoAppException(f"Could not import {module_name!r}.")
        else:
            return
    module = sys.modules[module_name]
    if app_name is None:
        return find_best_app(script_info, module)
    else:
        return find_app_by_string(script_info, module, app_name)
def get_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    import werkzeug
    from . import __version__
    click.echo(
        f"Python {platform.python_version()}\n"
        f"Flask {__version__}\n"
        f"Werkzeug {werkzeug.__version__}",
        color=ctx.color,
    )
    ctx.exit()
version_option = click.Option(
    ["--version"],
    help="Show the flask version",
    expose_value=False,
    callback=get_version,
    is_flag=True,
    is_eager=True,
)
class DispatchingApp:
    def __init__(self, loader, use_eager_loading=None):
        self.loader = loader
        self._app = None
        self._lock = Lock()
        self._bg_loading_exc_info = None
        if use_eager_loading is None:
            use_eager_loading = os.environ.get("WERKZEUG_RUN_MAIN") != "true"
        if use_eager_loading:
            self._load_unlocked()
        else:
            self._load_in_background()
    def _load_in_background(self):
        def _load_app():
            __traceback_hide__ = True  
            with self._lock:
                try:
                    self._load_unlocked()
                except Exception:
                    self._bg_loading_exc_info = sys.exc_info()
        t = Thread(target=_load_app, args=())
        t.start()
    def _flush_bg_loading_exception(self):
        __traceback_hide__ = True  
        exc_info = self._bg_loading_exc_info
        if exc_info is not None:
            self._bg_loading_exc_info = None
            raise exc_info
    def _load_unlocked(self):
        __traceback_hide__ = True  
        self._app = rv = self.loader()
        self._bg_loading_exc_info = None
        return rv
    def __call__(self, environ, start_response):
        __traceback_hide__ = True  
        if self._app is not None:
            return self._app(environ, start_response)
        self._flush_bg_loading_exception()
        with self._lock:
            if self._app is not None:
                rv = self._app
            else:
                rv = self._load_unlocked()
            return rv(environ, start_response)
class ScriptInfo:
    def __init__(self, app_import_path=None, create_app=None, set_debug_flag=True):
        self.app_import_path = app_import_path or os.environ.get("FLASK_APP")
        self.create_app = create_app
        self.data = {}
        self.set_debug_flag = set_debug_flag
        self._loaded_app = None
    def load_app(self):
        __traceback_hide__ = True  
        if self._loaded_app is not None:
            return self._loaded_app
        if self.create_app is not None:
            app = call_factory(self, self.create_app)
        else:
            if self.app_import_path:
                path, name = (
                    re.split(r":(?![\\/])", self.app_import_path, 1) + [None]
                )[:2]
                import_name = prepare_import(path)
                app = locate_app(self, import_name, name)
            else:
                for path in ("wsgi.py", "app.py"):
                    import_name = prepare_import(path)
                    app = locate_app(self, import_name, None, raise_if_not_found=False)
                    if app:
                        break
        if not app:
            raise NoAppException(
                "Could not locate a Flask application. You did not provide "
                'the "FLASK_APP" environment variable, and a "wsgi.py" or '
                '"app.py" module was not found in the current directory.'
            )
        if self.set_debug_flag:
            app.debug = get_debug_flag()
        self._loaded_app = app
        return app
pass_script_info = click.make_pass_decorator(ScriptInfo, ensure=True)
def with_appcontext(f):
    @click.pass_context
    def decorator(__ctx, *args, **kwargs):
        with __ctx.ensure_object(ScriptInfo).load_app().app_context():
            return __ctx.invoke(f, *args, **kwargs)
    return update_wrapper(decorator, f)
class AppGroup(click.Group):
    def command(self, *args, **kwargs):
        wrap_for_ctx = kwargs.pop("with_appcontext", True)
        def decorator(f):
            if wrap_for_ctx:
                f = with_appcontext(f)
            return click.Group.command(self, *args, **kwargs)(f)
        return decorator
    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", AppGroup)
        return click.Group.group(self, *args, **kwargs)
class FlaskGroup(AppGroup):
    def __init__(
        self,
        add_default_commands=True,
        create_app=None,
        add_version_option=True,
        load_dotenv=True,
        set_debug_flag=True,
        **extra,
    ):
        params = list(extra.pop("params", None) or ())
        if add_version_option:
            params.append(version_option)
        AppGroup.__init__(self, params=params, **extra)
        self.create_app = create_app
        self.load_dotenv = load_dotenv
        self.set_debug_flag = set_debug_flag
        if add_default_commands:
            self.add_command(run_command)
            self.add_command(shell_command)
            self.add_command(routes_command)
        self._loaded_plugin_commands = False
    def _load_plugin_commands(self):
        if self._loaded_plugin_commands:
            return
        try:
            import pkg_resources
        except ImportError:
            self._loaded_plugin_commands = True
            return
        for ep in pkg_resources.iter_entry_points("flask.commands"):
            self.add_command(ep.load(), ep.name)
        self._loaded_plugin_commands = True
    def get_command(self, ctx, name):
        self._load_plugin_commands()
        rv = super().get_command(ctx, name)
        if rv is not None:
            return rv
        info = ctx.ensure_object(ScriptInfo)
        try:
            return info.load_app().cli.get_command(ctx, name)
        except NoAppException as e:
            click.secho(f"Error: {e.format_message()}\n", err=True, fg="red")
    def list_commands(self, ctx):
        self._load_plugin_commands()
        rv = set(super().list_commands(ctx))
        info = ctx.ensure_object(ScriptInfo)
        try:
            rv.update(info.load_app().cli.list_commands(ctx))
        except NoAppException as e:
            click.secho(f"Error: {e.format_message()}\n", err=True, fg="red")
        except Exception:
            click.secho(f"{traceback.format_exc()}\n", err=True, fg="red")
        return sorted(rv)
    def main(self, *args, **kwargs):
        os.environ["FLASK_RUN_FROM_CLI"] = "true"
        if get_load_dotenv(self.load_dotenv):
            load_dotenv()
        obj = kwargs.get("obj")
        if obj is None:
            obj = ScriptInfo(
                create_app=self.create_app, set_debug_flag=self.set_debug_flag
            )
        kwargs["obj"] = obj
        kwargs.setdefault("auto_envvar_prefix", "FLASK")
        return super().main(*args, **kwargs)
def _path_is_ancestor(path, other):
    return os.path.join(path, other[len(path) :].lstrip(os.sep)) == other
def load_dotenv(path=None):
    if dotenv is None:
        if path or os.path.isfile(".env") or os.path.isfile(".flaskenv"):
            click.secho(
                " * Tip: There are .env or .flaskenv files present."
                ' Do "pip install python-dotenv" to use them.',
                fg="yellow",
                err=True,
            )
        return False
    if path is not None:
        if os.path.isfile(path):
            return dotenv.load_dotenv(path)
        return False
    new_dir = None
    for name in (".env", ".flaskenv"):
        path = dotenv.find_dotenv(name, usecwd=True)
        if not path:
            continue
        if new_dir is None:
            new_dir = os.path.dirname(path)
        dotenv.load_dotenv(path)
    return new_dir is not None  
def show_server_banner(env, debug, app_import_path, eager_loading):
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        return
    if app_import_path is not None:
        message = f" * Serving Flask app {app_import_path!r}"
        if not eager_loading:
            message += " (lazy loading)"
        click.echo(message)
    click.echo(f" * Environment: {env}")
    if env == "production":
        click.secho(
            "   WARNING: This is a development server. Do not use it in"
            " a production deployment.",
            fg="red",
        )
        click.secho("   Use a production WSGI server instead.", dim=True)
    if debug is not None:
        click.echo(f" * Debug mode: {'on' if debug else 'off'}")
class CertParamType(click.ParamType):
    name = "path"
    def __init__(self):
        self.path_type = click.Path(exists=True, dir_okay=False, resolve_path=True)
    def convert(self, value, param, ctx):
        if ssl is None:
            raise click.BadParameter(
                'Using "--cert" requires Python to be compiled with SSL support.',
                ctx,
                param,
            )
        try:
            return self.path_type(value, param, ctx)
        except click.BadParameter:
            value = click.STRING(value, param, ctx).lower()
            if value == "adhoc":
                try:
                    import cryptography  
                except ImportError:
                    raise click.BadParameter(
                        "Using ad-hoc certificates requires the cryptography library.",
                        ctx,
                        param,
                    )
                return value
            obj = import_string(value, silent=True)
            if isinstance(obj, ssl.SSLContext):
                return obj
            raise
def _validate_key(ctx, param, value):
    cert = ctx.params.get("cert")
    is_adhoc = cert == "adhoc"
    is_context = ssl and isinstance(cert, ssl.SSLContext)
    if value is not None:
        if is_adhoc:
            raise click.BadParameter(
                'When "--cert" is "adhoc", "--key" is not used.', ctx, param
            )
        if is_context:
            raise click.BadParameter(
                'When "--cert" is an SSLContext object, "--key is not used.', ctx, param
            )
        if not cert:
            raise click.BadParameter('"--cert" must also be specified.', ctx, param)
        ctx.params["cert"] = cert, value
    else:
        if cert and not (is_adhoc or is_context):
            raise click.BadParameter('Required when using "--cert".', ctx, param)
    return value
class SeparatedPathType(click.Path):
    def convert(self, value, param, ctx):
        items = self.split_envvar_value(value)
        super_convert = super().convert
        return [super_convert(item, param, ctx) for item in items]
@click.command("run", short_help="Run a development server.")
@click.option("--host", "-h", default="127.0.0.1", help="The interface to bind to.")
@click.option("--port", "-p", default=5000, help="The port to bind to.")
@click.option(
    "--cert", type=CertParamType(), help="Specify a certificate file to use HTTPS."
)
@click.option(
    "--key",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    callback=_validate_key,
    expose_value=False,
    help="The key file to use when specifying a certificate.",
)
@click.option(
    "--reload/--no-reload",
    default=None,
    help="Enable or disable the reloader. By default the reloader "
    "is active if debug is enabled.",
)
@click.option(
    "--debugger/--no-debugger",
    default=None,
    help="Enable or disable the debugger. By default the debugger "
    "is active if debug is enabled.",
)
@click.option(
    "--eager-loading/--lazy-loading",
    default=None,
    help="Enable or disable eager loading. By default eager "
    "loading is enabled if the reloader is disabled.",
)
@click.option(
    "--with-threads/--without-threads",
    default=True,
    help="Enable or disable multithreading.",
)
@click.option(
    "--extra-files",
    default=None,
    type=SeparatedPathType(),
    help=(
        "Extra files that trigger a reload on change. Multiple paths"
        f" are separated by {os.path.pathsep!r}."
    ),
)
@pass_script_info
def run_command(
    info, host, port, reload, debugger, eager_loading, with_threads, cert, extra_files
):
    debug = get_debug_flag()
    if reload is None:
        reload = debug
    if debugger is None:
        debugger = debug
    show_server_banner(get_env(), debug, info.app_import_path, eager_loading)
    app = DispatchingApp(info.load_app, use_eager_loading=eager_loading)
    from werkzeug.serving import run_simple
    run_simple(
        host,
        port,
        app,
        use_reloader=reload,
        use_debugger=debugger,
        threaded=with_threads,
        ssl_context=cert,
        extra_files=extra_files,
    )
@click.command("shell", short_help="Run a shell in the app context.")
@with_appcontext
def shell_command():
    import code
    from .globals import _app_ctx_stack
    app = _app_ctx_stack.top.app
    banner = (
        f"Python {sys.version} on {sys.platform}\n"
        f"App: {app.import_name} [{app.env}]\n"
        f"Instance: {app.instance_path}"
    )
    ctx = {}
    startup = os.environ.get("PYTHONSTARTUP")
    if startup and os.path.isfile(startup):
        with open(startup) as f:
            eval(compile(f.read(), startup, "exec"), ctx)
    ctx.update(app.make_shell_context())
    code.interact(banner=banner, local=ctx)
@click.command("routes", short_help="Show the routes for the app.")
@click.option(
    "--sort",
    "-s",
    type=click.Choice(("endpoint", "methods", "rule", "match")),
    default="endpoint",
    help=(
        'Method to sort routes by. "match" is the order that Flask will match '
        "routes when dispatching a request."
    ),
)
@click.option("--all-methods", is_flag=True, help="Show HEAD and OPTIONS methods.")
@with_appcontext
def routes_command(sort, all_methods):
    rules = list(current_app.url_map.iter_rules())
    if not rules:
        click.echo("No routes were registered.")
        return
    ignored_methods = set(() if all_methods else ("HEAD", "OPTIONS"))
    if sort in ("endpoint", "rule"):
        rules = sorted(rules, key=attrgetter(sort))
    elif sort == "methods":
        rules = sorted(rules, key=lambda rule: sorted(rule.methods))
    rule_methods = [", ".join(sorted(rule.methods - ignored_methods)) for rule in rules]
    headers = ("Endpoint", "Methods", "Rule")
    widths = (
        max(len(rule.endpoint) for rule in rules),
        max(len(methods) for methods in rule_methods),
        max(len(rule.rule) for rule in rules),
    )
    widths = [max(len(h), w) for h, w in zip(headers, widths)]
    row = "{{0:<{0}}}  {{1:<{1}}}  {{2:<{2}}}".format(*widths)
    click.echo(row.format(*headers).strip())
    click.echo(row.format(*("-" * width for width in widths)))
    for rule, methods in zip(rules, rule_methods):
        click.echo(row.format(rule.endpoint, methods, rule.rule).rstrip())
cli = FlaskGroup(
    help="""\
A general utility script for Flask applications.
Provides commands from Flask, extensions, and the application. Loads the
application defined in the FLASK_APP environment variable, or from a wsgi.py
file. Setting the FLASK_ENV environment variable to 'development' will enable
debug mode.
\b
  {prefix}{cmd} FLASK_APP=hello.py
  {prefix}{cmd} FLASK_ENV=development
  {prefix}flask run
""".format(
        cmd="export" if os.name == "posix" else "set",
        prefix="$ " if os.name == "posix" else "> ",
    )
)
def main(as_module=False):
    cli.main(args=sys.argv[1:], prog_name="python -m flask" if as_module else None)
if __name__ == "__main__":
    main(as_module=True)