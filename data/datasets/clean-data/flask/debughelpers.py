import os
from warnings import warn
from .app import Flask
from .blueprints import Blueprint
from .globals import _request_ctx_stack
class UnexpectedUnicodeError(AssertionError, UnicodeError):
class DebugFilesKeyError(KeyError, AssertionError):
    def __init__(self, request, key):
        form_matches = request.form.getlist(key)
        buf = [
            f"You tried to access the file {key!r} in the request.files"
            " dictionary but it does not exist. The mimetype for the"
            f" request is {request.mimetype!r} instead of"
            " 'multipart/form-data' which means that no file contents"
            " were transmitted. To fix this error you should provide"
            ' enctype="multipart/form-data" in your form.'
        ]
        if form_matches:
            names = ", ".join(repr(x) for x in form_matches)
            buf.append(
                "\n\nThe browser instead transmitted some file names. "
                f"This was submitted: {names}"
            )
        self.msg = "".join(buf)
    def __str__(self):
        return self.msg
class FormDataRoutingRedirect(AssertionError):
    def __init__(self, request):
        exc = request.routing_exception
        buf = [
            f"A request was sent to this URL ({request.url}) but a"
            " redirect was issued automatically by the routing system"
            f" to {exc.new_url!r}."
        ]
        if f"{request.base_url}/" == exc.new_url.split("?")[0]:
            buf.append(
                "  The URL was defined with a trailing slash so Flask"
                " will automatically redirect to the URL with the"
                " trailing slash if it was accessed without one."
            )
        buf.append(
            "  Make sure to directly send your"
            f" {request.method}-request to this URL since we can't make"
            " browsers or HTTP clients redirect with form data reliably"
            " or without user interaction."
        )
        buf.append("\n\nNote: this exception is only raised in debug mode")
        AssertionError.__init__(self, "".join(buf).encode("utf-8"))
def attach_enctype_error_multidict(request):
    oldcls = request.files.__class__
    class newcls(oldcls):
        def __getitem__(self, key):
            try:
                return oldcls.__getitem__(self, key)
            except KeyError:
                if key not in request.form:
                    raise
                raise DebugFilesKeyError(request, key)
    newcls.__name__ = oldcls.__name__
    newcls.__module__ = oldcls.__module__
    request.files.__class__ = newcls
def _dump_loader_info(loader):
    yield f"class: {type(loader).__module__}.{type(loader).__name__}"
    for key, value in sorted(loader.__dict__.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, (tuple, list)):
            if not all(isinstance(x, str) for x in value):
                continue
            yield f"{key}:"
            for item in value:
                yield f"  - {item}"
            continue
        elif not isinstance(value, (str, int, float, bool)):
            continue
        yield f"{key}: {value!r}"
def explain_template_loading_attempts(app, template, attempts):
    info = [f"Locating template {template!r}:"]
    total_found = 0
    blueprint = None
    reqctx = _request_ctx_stack.top
    if reqctx is not None and reqctx.request.blueprint is not None:
        blueprint = reqctx.request.blueprint
    for idx, (loader, srcobj, triple) in enumerate(attempts):
        if isinstance(srcobj, Flask):
            src_info = f"application {srcobj.import_name!r}"
        elif isinstance(srcobj, Blueprint):
            src_info = f"blueprint {srcobj.name!r} ({srcobj.import_name})"
        else:
            src_info = repr(srcobj)
        info.append(f"{idx + 1:5}: trying loader of {src_info}")
        for line in _dump_loader_info(loader):
            info.append(f"       {line}")
        if triple is None:
            detail = "no match"
        else:
            detail = f"found ({triple[1] or '<string>'!r})"
            total_found += 1
        info.append(f"       -> {detail}")
    seems_fishy = False
    if total_found == 0:
        info.append("Error: the template could not be found.")
        seems_fishy = True
    elif total_found > 1:
        info.append("Warning: multiple loaders returned a match for the template.")
        seems_fishy = True
    if blueprint is not None and seems_fishy:
        info.append(
            "  The template was looked up from an endpoint that belongs"
            f" to the blueprint {blueprint!r}."
        )
        info.append("  Maybe you did not place a template in the right folder?")
        info.append("  See https://flask.palletsprojects.com/blueprints/#templates")
    app.logger.info("\n".join(info))
def explain_ignored_app_run():
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        warn(
            Warning(
                "Silently ignoring app.run() because the application is"
                " run from the flask command line executable. Consider"
                ' putting app.run() behind an if __name__ == "__main__"'
                " guard to silence this warning."
            ),
            stacklevel=3,
        )