from functools import update_wrapper
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
class BlueprintSetupState:
    def __init__(self, blueprint, app, options, first_registration):
        self.app = app
        self.blueprint = blueprint
        self.options = options
        self.first_registration = first_registration
        subdomain = self.options.get("subdomain")
        if subdomain is None:
            subdomain = self.blueprint.subdomain
        self.subdomain = subdomain
        url_prefix = self.options.get("url_prefix")
        if url_prefix is None:
            url_prefix = self.blueprint.url_prefix
        self.url_prefix = url_prefix
        self.url_defaults = dict(self.blueprint.url_values_defaults)
        self.url_defaults.update(self.options.get("url_defaults", ()))
    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        if self.url_prefix is not None:
            if rule:
                rule = "/".join((self.url_prefix.rstrip("/"), rule.lstrip("/")))
            else:
                rule = self.url_prefix
        options.setdefault("subdomain", self.subdomain)
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)
        defaults = self.url_defaults
        if "defaults" in options:
            defaults = dict(defaults, **options.pop("defaults"))
        self.app.add_url_rule(
            rule,
            f"{self.blueprint.name}.{endpoint}",
            view_func,
            defaults=defaults,
            **options,
        )
class Blueprint(Scaffold):
    warn_on_modifications = False
    _got_registered_once = False
    json_encoder = None
    json_decoder = None
    import_name = None
    template_folder = None
    root_path = None
    def __init__(
        self,
        name,
        import_name,
        static_folder=None,
        static_url_path=None,
        template_folder=None,
        url_prefix=None,
        subdomain=None,
        url_defaults=None,
        root_path=None,
        cli_group=_sentinel,
    ):
        super().__init__(
            import_name=import_name,
            static_folder=static_folder,
            static_url_path=static_url_path,
            template_folder=template_folder,
            root_path=root_path,
        )
        self.name = name
        self.url_prefix = url_prefix
        self.subdomain = subdomain
        self.deferred_functions = []
        if url_defaults is None:
            url_defaults = {}
        self.url_values_defaults = url_defaults
        self.cli_group = cli_group
    def _is_setup_finished(self):
        return self.warn_on_modifications and self._got_registered_once
    def record(self, func):
        if self._got_registered_once and self.warn_on_modifications:
            from warnings import warn
            warn(
                Warning(
                    "The blueprint was already registered once "
                    "but is getting modified now.  These changes "
                    "will not show up."
                )
            )
        self.deferred_functions.append(func)
    def record_once(self, func):
        def wrapper(state):
            if state.first_registration:
                func(state)
        return self.record(update_wrapper(wrapper, func))
    def make_setup_state(self, app, options, first_registration=False):
        return BlueprintSetupState(self, app, options, first_registration)
    def register(self, app, options, first_registration=False):
        self._got_registered_once = True
        state = self.make_setup_state(app, options, first_registration)
        if self.has_static_folder:
            state.add_url_rule(
                f"{self.static_url_path}/<path:filename>",
                view_func=self.send_static_file,
                endpoint="static",
            )
        def merge_dict_lists(self_dict, app_dict):
            for key, values in self_dict.items():
                key = self.name if key is None else f"{self.name}.{key}"
                app_dict.setdefault(key, []).extend(values)
        def merge_dict_nested(self_dict, app_dict):
            for key, value in self_dict.items():
                key = self.name if key is None else f"{self.name}.{key}"
                app_dict[key] = value
        app.view_functions.update(self.view_functions)
        merge_dict_lists(self.before_request_funcs, app.before_request_funcs)
        merge_dict_lists(self.after_request_funcs, app.after_request_funcs)
        merge_dict_lists(self.teardown_request_funcs, app.teardown_request_funcs)
        merge_dict_lists(self.url_default_functions, app.url_default_functions)
        merge_dict_lists(self.url_value_preprocessors, app.url_value_preprocessors)
        merge_dict_lists(
            self.template_context_processors, app.template_context_processors
        )
        merge_dict_nested(self.error_handler_spec, app.error_handler_spec)
        for deferred in self.deferred_functions:
            deferred(state)
        cli_resolved_group = options.get("cli_group", self.cli_group)
        if not self.cli.commands:
            return
        if cli_resolved_group is None:
            app.cli.commands.update(self.cli.commands)
        elif cli_resolved_group is _sentinel:
            self.cli.name = self.name
            app.cli.add_command(self.cli)
        else:
            self.cli.name = cli_resolved_group
            app.cli.add_command(self.cli)
    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        if endpoint:
            assert "." not in endpoint, "Blueprint endpoints should not contain dots"
        if view_func and hasattr(view_func, "__name__"):
            assert (
                "." not in view_func.__name__
            ), "Blueprint view function name should not contain dots"
        self.record(lambda s: s.add_url_rule(rule, endpoint, view_func, **options))
    def app_template_filter(self, name=None):
        def decorator(f):
            self.add_app_template_filter(f, name=name)
            return f
        return decorator
    def add_app_template_filter(self, f, name=None):
        def register_template(state):
            state.app.jinja_env.filters[name or f.__name__] = f
        self.record_once(register_template)
    def app_template_test(self, name=None):
        def decorator(f):
            self.add_app_template_test(f, name=name)
            return f
        return decorator
    def add_app_template_test(self, f, name=None):
        def register_template(state):
            state.app.jinja_env.tests[name or f.__name__] = f
        self.record_once(register_template)
    def app_template_global(self, name=None):
        def decorator(f):
            self.add_app_template_global(f, name=name)
            return f
        return decorator
    def add_app_template_global(self, f, name=None):
        def register_template(state):
            state.app.jinja_env.globals[name or f.__name__] = f
        self.record_once(register_template)
    def before_app_request(self, f):
        self.record_once(
            lambda s: s.app.before_request_funcs.setdefault(None, []).append(f)
        )
        return f
    def before_app_first_request(self, f):
        self.record_once(lambda s: s.app.before_first_request_funcs.append(f))
        return f
    def after_app_request(self, f):
        self.record_once(
            lambda s: s.app.after_request_funcs.setdefault(None, []).append(f)
        )
        return f
    def teardown_app_request(self, f):
        self.record_once(
            lambda s: s.app.teardown_request_funcs.setdefault(None, []).append(f)
        )
        return f
    def app_context_processor(self, f):
        self.record_once(
            lambda s: s.app.template_context_processors.setdefault(None, []).append(f)
        )
        return f
    def app_errorhandler(self, code):
        def decorator(f):
            self.record_once(lambda s: s.app.errorhandler(code)(f))
            return f
        return decorator
    def app_url_value_preprocessor(self, f):
        self.record_once(
            lambda s: s.app.url_value_preprocessors.setdefault(None, []).append(f)
        )
        return f
    def app_url_defaults(self, f):
        self.record_once(
            lambda s: s.app.url_default_functions.setdefault(None, []).append(f)
        )
        return f