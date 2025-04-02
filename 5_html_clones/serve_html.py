import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = "output"

os.chdir(DIRECTORY)

Handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()
