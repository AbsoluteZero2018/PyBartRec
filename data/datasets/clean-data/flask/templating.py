from jinja2 import BaseLoader
from jinja2 import Environment as BaseEnvironment
from jinja2 import TemplateNotFound
from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .signals import before_render_template
from .signals import template_rendered
def _default_template_ctx_processor():
    reqctx = _request_ctx_stack.top
    appctx = _app_ctx_stack.top
    rv = {}
    if appctx is not None:
        rv["g"] = appctx.g
    if reqctx is not None:
        rv["request"] = reqctx.request
        rv["session"] = reqctx.session
    return rv
class Environment(BaseEnvironment):
    def __init__(self, app, **options):
        if "loader" not in options:
            options["loader"] = app.create_global_jinja_loader()
        BaseEnvironment.__init__(self, **options)
        self.app = app
class DispatchingJinjaLoader(BaseLoader):
    def __init__(self, app):
        self.app = app
    def get_source(self, environment, template):
        if self.app.config["EXPLAIN_TEMPLATE_LOADING"]:
            return self._get_source_explained(environment, template)
        return self._get_source_fast(environment, template)
    def _get_source_explained(self, environment, template):
        attempts = []
        trv = None
        for srcobj, loader in self._iter_loaders(template):
            try:
                rv = loader.get_source(environment, template)
                if trv is None:
                    trv = rv
            except TemplateNotFound:
                rv = None
            attempts.append((loader, srcobj, rv))
        from .debughelpers import explain_template_loading_attempts
        explain_template_loading_attempts(self.app, template, attempts)
        if trv is not None:
            return trv
        raise TemplateNotFound(template)
    def _get_source_fast(self, environment, template):
        for _srcobj, loader in self._iter_loaders(template):
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                continue
        raise TemplateNotFound(template)
    def _iter_loaders(self, template):
        loader = self.app.jinja_loader
        if loader is not None:
            yield self.app, loader
        for blueprint in self.app.iter_blueprints():
            loader = blueprint.jinja_loader
            if loader is not None:
                yield blueprint, loader
    def list_templates(self):
        result = set()
        loader = self.app.jinja_loader
        if loader is not None:
            result.update(loader.list_templates())
        for blueprint in self.app.iter_blueprints():
            loader = blueprint.jinja_loader
            if loader is not None:
                for template in loader.list_templates():
                    result.add(template)
        return list(result)
def _render(template, context, app):
    before_render_template.send(app, template=template, context=context)
    rv = template.render(context)
    template_rendered.send(app, template=template, context=context)
    return rv
def render_template(template_name_or_list, **context):
    ctx = _app_ctx_stack.top
    ctx.app.update_template_context(context)
    return _render(
        ctx.app.jinja_env.get_or_select_template(template_name_or_list),
        context,
        ctx.app,
    )
def render_template_string(source, **context):
    ctx = _app_ctx_stack.top
    ctx.app.update_template_context(context)
    return _render(ctx.app.jinja_env.from_string(source), context, ctx.app)