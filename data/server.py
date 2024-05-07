from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    handler_class = partial(
        CORSHTTPRequestHandler,
        directory="xxxxxx",
    )
    server = HTTPServer(("localhost", 8080), handler_class)
    print("Serving at http://localhost:8080")
    server.serve_forever()
