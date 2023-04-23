from werkzeug.exceptions import BadRequest
from werkzeug.wrappers import Request as RequestBase
from werkzeug.wrappers import Response as ResponseBase
from werkzeug.wrappers.json import JSONMixin as _JSONMixin
from . import json
from .globals import current_app
class JSONMixin(_JSONMixin):
    json_module = json
    def on_json_loading_failed(self, e):
        if current_app and current_app.debug:
            raise BadRequest(f"Failed to decode JSON object: {e}")
        raise BadRequest()
class Request(RequestBase, JSONMixin):
    url_rule = None
    view_args = None
    routing_exception = None
    @property
    def max_content_length(self):
        if current_app:
            return current_app.config["MAX_CONTENT_LENGTH"]
    @property
    def endpoint(self):
        if self.url_rule is not None:
            return self.url_rule.endpoint
    @property
    def blueprint(self):
        if self.url_rule and "." in self.url_rule.endpoint:
            return self.url_rule.endpoint.rsplit(".", 1)[0]
    def _load_form_data(self):
        RequestBase._load_form_data(self)
        if (
            current_app
            and current_app.debug
            and self.mimetype != "multipart/form-data"
            and not self.files
        ):
            from .debughelpers import attach_enctype_error_multidict
            attach_enctype_error_multidict(self)
class Response(ResponseBase, JSONMixin):
    default_mimetype = "text/html"
    def _get_data_for_json(self, cache):
        return self.get_data()
    @property
    def max_cookie_size(self):
        if current_app:
            return current_app.config["MAX_COOKIE_SIZE"]
        return super().max_cookie_size