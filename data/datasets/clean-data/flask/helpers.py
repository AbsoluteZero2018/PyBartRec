import io
import mimetypes
import os
import pkgutil
import posixpath
import socket
import sys
import unicodedata
from functools import update_wrapper
from threading import RLock
from time import time
from zlib import adler32
from jinja2 import FileSystemLoader
from werkzeug.datastructures import Headers
from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import NotFound
from werkzeug.exceptions import RequestedRangeNotSatisfiable
from werkzeug.routing import BuildError
from werkzeug.urls import url_quote
from werkzeug.wsgi import wrap_file
from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .globals import current_app
from .globals import request
from .globals import session
from .signals import message_flashed
_missing = object()
_os_alt_seps = list(
    sep for sep in [os.path.sep, os.path.altsep] if sep not in (None, "/")
)
def get_env():
    return os.environ.get("FLASK_ENV") or "production"
def get_debug_flag():
    val = os.environ.get("FLASK_DEBUG")
    if not val:
        return get_env() == "development"
    return val.lower() not in ("0", "false", "no")
def get_load_dotenv(default=True):
    val = os.environ.get("FLASK_SKIP_DOTENV")
    if not val:
        return default
    return val.lower() in ("0", "false", "no")
def stream_with_context(generator_or_function):
    try:
        gen = iter(generator_or_function)
    except TypeError:
        def decorator(*args, **kwargs):
            gen = generator_or_function(*args, **kwargs)
            return stream_with_context(gen)
        return update_wrapper(decorator, generator_or_function)
    def generator():
        ctx = _request_ctx_stack.top
        if ctx is None:
            raise RuntimeError(
                "Attempted to stream with context but "
                "there was no context in the first place to keep around."
            )
        with ctx:
            yield None
            try:
                yield from gen
            finally:
                if hasattr(gen, "close"):
                    gen.close()
    wrapped_g = generator()
    next(wrapped_g)
    return wrapped_g
def make_response(*args):
    if not args:
        return current_app.response_class()
    if len(args) == 1:
        args = args[0]
    return current_app.make_response(args)
def url_for(endpoint, **values):
    appctx = _app_ctx_stack.top
    reqctx = _request_ctx_stack.top
    if appctx is None:
        raise RuntimeError(
            "Attempted to generate a URL without the application context being"
            " pushed. This has to be executed when application context is"
            " available."
        )
    if reqctx is not None:
        url_adapter = reqctx.url_adapter
        blueprint_name = request.blueprint
        if endpoint[:1] == ".":
            if blueprint_name is not None:
                endpoint = f"{blueprint_name}{endpoint}"
            else:
                endpoint = endpoint[1:]
        external = values.pop("_external", False)
    else:
        url_adapter = appctx.url_adapter
        if url_adapter is None:
            raise RuntimeError(
                "Application was not able to create a URL adapter for request"
                " independent URL generation. You might be able to fix this by"
                " setting the SERVER_NAME config variable."
            )
        external = values.pop("_external", True)
    anchor = values.pop("_anchor", None)
    method = values.pop("_method", None)
    scheme = values.pop("_scheme", None)
    appctx.app.inject_url_defaults(endpoint, values)
    old_scheme = None
    if scheme is not None:
        if not external:
            raise ValueError("When specifying _scheme, _external must be True")
        old_scheme = url_adapter.url_scheme
        url_adapter.url_scheme = scheme
    try:
        try:
            rv = url_adapter.build(
                endpoint, values, method=method, force_external=external
            )
        finally:
            if old_scheme is not None:
                url_adapter.url_scheme = old_scheme
    except BuildError as error:
        values["_external"] = external
        values["_anchor"] = anchor
        values["_method"] = method
        values["_scheme"] = scheme
        return appctx.app.handle_url_build_error(error, endpoint, values)
    if anchor is not None:
        rv += f"#{url_quote(anchor)}"
    return rv
def get_template_attribute(template_name, attribute):
    return getattr(current_app.jinja_env.get_template(template_name).module, attribute)
def flash(message, category="message"):
    flashes = session.get("_flashes", [])
    flashes.append((category, message))
    session["_flashes"] = flashes
    message_flashed.send(
        current_app._get_current_object(), message=message, category=category
    )
def get_flashed_messages(with_categories=False, category_filter=()):
    flashes = _request_ctx_stack.top.flashes
    if flashes is None:
        _request_ctx_stack.top.flashes = flashes = (
            session.pop("_flashes") if "_flashes" in session else []
        )
    if category_filter:
        flashes = list(filter(lambda f: f[0] in category_filter, flashes))
    if not with_categories:
        return [x[1] for x in flashes]
    return flashes
def send_file(
    filename_or_fp,
    mimetype=None,
    as_attachment=False,
    attachment_filename=None,
    add_etags=True,
    cache_timeout=None,
    conditional=False,
    last_modified=None,
):
    mtime = None
    fsize = None
    if hasattr(filename_or_fp, "__fspath__"):
        filename_or_fp = os.fspath(filename_or_fp)
    if isinstance(filename_or_fp, str):
        filename = filename_or_fp
        if not os.path.isabs(filename):
            filename = os.path.join(current_app.root_path, filename)
        file = None
        if attachment_filename is None:
            attachment_filename = os.path.basename(filename)
    else:
        file = filename_or_fp
        filename = None
    if mimetype is None:
        if attachment_filename is not None:
            mimetype = (
                mimetypes.guess_type(attachment_filename)[0]
                or "application/octet-stream"
            )
        if mimetype is None:
            raise ValueError(
                "Unable to infer MIME-type because no filename is available. "
                "Please set either `attachment_filename`, pass a filepath to "
                "`filename_or_fp` or set your own MIME-type via `mimetype`."
            )
    headers = Headers()
    if as_attachment:
        if attachment_filename is None:
            raise TypeError("filename unavailable, required for sending as attachment")
        if not isinstance(attachment_filename, str):
            attachment_filename = attachment_filename.decode("utf-8")
        try:
            attachment_filename = attachment_filename.encode("ascii")
        except UnicodeEncodeError:
            quoted = url_quote(attachment_filename, safe="")
            filenames = {
                "filename": unicodedata.normalize("NFKD", attachment_filename).encode(
                    "ascii", "ignore"
                ),
                "filename*": f"UTF-8''{quoted}",
            }
        else:
            filenames = {"filename": attachment_filename}
        headers.add("Content-Disposition", "attachment", **filenames)
    if current_app.use_x_sendfile and filename:
        if file is not None:
            file.close()
        headers["X-Sendfile"] = filename
        fsize = os.path.getsize(filename)
        data = None
    else:
        if file is None:
            file = open(filename, "rb")
            mtime = os.path.getmtime(filename)
            fsize = os.path.getsize(filename)
        elif isinstance(file, io.BytesIO):
            fsize = file.getbuffer().nbytes
        elif isinstance(file, io.TextIOBase):
            raise ValueError("Files must be opened in binary mode or use BytesIO.")
        data = wrap_file(request.environ, file)
    if fsize is not None:
        headers["Content-Length"] = fsize
    rv = current_app.response_class(
        data, mimetype=mimetype, headers=headers, direct_passthrough=True
    )
    if last_modified is not None:
        rv.last_modified = last_modified
    elif mtime is not None:
        rv.last_modified = mtime
    rv.cache_control.public = True
    if cache_timeout is None:
        cache_timeout = current_app.get_send_file_max_age(filename)
    if cache_timeout is not None:
        rv.cache_control.max_age = cache_timeout
        rv.expires = int(time() + cache_timeout)
    if add_etags and filename is not None:
        from warnings import warn
        try:
            check = (
                adler32(
                    filename.encode("utf-8") if isinstance(filename, str) else filename
                )
                & 0xFFFFFFFF
            )
            rv.set_etag(
                f"{os.path.getmtime(filename)}-{os.path.getsize(filename)}-{check}"
            )
        except OSError:
            warn(
                f"Access {filename} failed, maybe it does not exist, so"
                " ignore etags in headers",
                stacklevel=2,
            )
    if conditional:
        try:
            rv = rv.make_conditional(request, accept_ranges=True, complete_length=fsize)
        except RequestedRangeNotSatisfiable:
            if file is not None:
                file.close()
            raise
        if rv.status_code == 304:
            rv.headers.pop("x-sendfile", None)
    return rv
def safe_join(directory, *pathnames):
    parts = [directory]
    for filename in pathnames:
        if filename != "":
            filename = posixpath.normpath(filename)
        if (
            any(sep in filename for sep in _os_alt_seps)
            or os.path.isabs(filename)
            or filename == ".."
            or filename.startswith("../")
        ):
            raise NotFound()
        parts.append(filename)
    return posixpath.join(*parts)
def send_from_directory(directory, filename, **options):
    filename = os.fspath(filename)
    directory = os.fspath(directory)
    filename = safe_join(directory, filename)
    if not os.path.isabs(filename):
        filename = os.path.join(current_app.root_path, filename)
    try:
        if not os.path.isfile(filename):
            raise NotFound()
    except (TypeError, ValueError):
        raise BadRequest()
    options.setdefault("conditional", True)
    return send_file(filename, **options)
def get_root_path(import_name):
    mod = sys.modules.get(import_name)
    if mod is not None and hasattr(mod, "__file__"):
        return os.path.dirname(os.path.abspath(mod.__file__))
    loader = pkgutil.get_loader(import_name)
    if loader is None or import_name == "__main__":
        return os.getcwd()
    if hasattr(loader, "get_filename"):
        filepath = loader.get_filename(import_name)
    else:
        __import__(import_name)
        mod = sys.modules[import_name]
        filepath = getattr(mod, "__file__", None)
        if filepath is None:
            raise RuntimeError(
                "No root path can be found for the provided module"
                f" {import_name!r}. This can happen because the module"
                " came from an import hook that does not provide file"
                " name information or because it's a namespace package."
                " In this case the root path needs to be explicitly"
                " provided."
            )
    return os.path.dirname(os.path.abspath(filepath))
def _matching_loader_thinks_module_is_package(loader, mod_name):
    cls = type(loader)
    if hasattr(loader, "is_package"):
        return loader.is_package(mod_name)
    elif cls.__module__ == "_frozen_importlib" and cls.__name__ == "NamespaceLoader":
        return True
    raise AttributeError(
        f"{cls.__name__}.is_package() method is missing but is required"
        " for PEP 302 import hooks."
    )
def _find_package_path(root_mod_name):
    import importlib.util
    try:
        spec = importlib.util.find_spec(root_mod_name)
        if spec is None:
            raise ValueError("not found")
    except (ImportError, ValueError):
        pass  
    else:
        if spec.origin in {"namespace", None}:
            return os.path.dirname(next(iter(spec.submodule_search_locations)))
        elif spec.submodule_search_locations:
            return os.path.dirname(os.path.dirname(spec.origin))
        else:
            return os.path.dirname(spec.origin)
    loader = pkgutil.get_loader(root_mod_name)
    if loader is None or root_mod_name == "__main__":
        return os.getcwd()
    else:
        if hasattr(loader, "get_filename"):
            filename = loader.get_filename(root_mod_name)
        elif hasattr(loader, "archive"):
            filename = loader.archive
        else:
            __import__(root_mod_name)
            filename = sys.modules[root_mod_name].__file__
        package_path = os.path.abspath(os.path.dirname(filename))
        if _matching_loader_thinks_module_is_package(loader, root_mod_name):
            package_path = os.path.dirname(package_path)
    return package_path
def find_package(import_name):
    root_mod_name, _, _ = import_name.partition(".")
    package_path = _find_package_path(root_mod_name)
    site_parent, site_folder = os.path.split(package_path)
    py_prefix = os.path.abspath(sys.prefix)
    if package_path.startswith(py_prefix):
        return py_prefix, package_path
    elif site_folder.lower() == "site-packages":
        parent, folder = os.path.split(site_parent)
        if folder.lower() == "lib":
            base_dir = parent
        elif os.path.basename(parent).lower() == "lib":
            base_dir = os.path.dirname(parent)
        else:
            base_dir = site_parent
        return base_dir, package_path
    return None, package_path
class locked_cached_property:
    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func
        self.lock = RLock()
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        with self.lock:
            value = obj.__dict__.get(self.__name__, _missing)
            if value is _missing:
                value = self.func(obj)
                obj.__dict__[self.__name__] = value
            return value
class _PackageBoundObject:
    import_name = None
    template_folder = None
    root_path = None
    def __init__(self, import_name, template_folder=None, root_path=None):
        self.import_name = import_name
        self.template_folder = template_folder
        if root_path is None:
            root_path = get_root_path(self.import_name)
        self.root_path = root_path
        self._static_folder = None
        self._static_url_path = None
        from .cli import AppGroup
        self.cli = AppGroup()
    @property
    def static_folder(self):
        if self._static_folder is not None:
            return os.path.join(self.root_path, self._static_folder)
    @static_folder.setter
    def static_folder(self, value):
        if value is not None:
            value = os.fspath(value).rstrip(r"\/")
        self._static_folder = value
    @property
    def static_url_path(self):
        if self._static_url_path is not None:
            return self._static_url_path
        if self.static_folder is not None:
            basename = os.path.basename(self.static_folder)
            return f"/{basename}".rstrip("/")
    @static_url_path.setter
    def static_url_path(self, value):
        if value is not None:
            value = value.rstrip("/")
        self._static_url_path = value
    @property
    def has_static_folder(self):
        return self.static_folder is not None
    @locked_cached_property
    def jinja_loader(self):
        if self.template_folder is not None:
            return FileSystemLoader(os.path.join(self.root_path, self.template_folder))
    def get_send_file_max_age(self, filename):
        return total_seconds(current_app.send_file_max_age_default)
    def send_static_file(self, filename):
        if not self.has_static_folder:
            raise RuntimeError("No static folder for this object")
        cache_timeout = self.get_send_file_max_age(filename)
        return send_from_directory(
            self.static_folder, filename, cache_timeout=cache_timeout
        )
    def open_resource(self, resource, mode="rb"):
        if mode not in {"r", "rt", "rb"}:
            raise ValueError("Resources can only be opened for reading")
        return open(os.path.join(self.root_path, resource), mode)
def total_seconds(td):
    return td.days * 60 * 60 * 24 + td.seconds
def is_ip(value):
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, value)
        except OSError:
            pass
        else:
            return True
    return False