/* Static demo: serves the payloads captured from a real JobScope run instead of
   the FastAPI backend. Everything lives in fixtures.js; nothing leaves the page. */
(function () {
  var F = window.JS_DEMO;
  if (!F) return;

  var realFetch = window.fetch ? window.fetch.bind(window) : null;

  function reply(obj, status) {
    return new Response(JSON.stringify(obj), {
      status: status || 200,
      headers: { "Content-Type": "application/json" }
    });
  }

  function replay(frames) {
    var enc = new TextEncoder();
    return new Response(new ReadableStream({
      start: function (controller) {
        var i = 0;
        (function step() {
          if (i >= frames.length) { controller.close(); return; }
          var f = frames[i++];
          controller.enqueue(enc.encode("data: " + JSON.stringify(f.payload) + "\n\n"));
          setTimeout(step, f.wait || 200);
        })();
      }
    }), { headers: { "Content-Type": "text/event-stream" } });
  }

  var NOT_HERE = "This static demo has no backend — that feature needs the local app.";

  window.fetch = function (input, opts) {
    var url = (typeof input === "string" ? input : (input && input.url) || "");
    var at = url.indexOf("/api/");
    if (at < 0) return realFetch ? realFetch(input, opts) : Promise.reject(new Error("offline"));
    var path = url.slice(at).split("?")[0];
    var method = ((opts && opts.method) || "GET").toUpperCase();

    if (path === "/api/auth/state") return Promise.resolve(reply(F.session));
    if (path === "/api/auth/login" || path === "/api/auth/register") return Promise.resolve(reply(F.session));
    if (path === "/api/auth/session" || path === "/api/auth/extend")
      return Promise.resolve(reply({ ok: true, seconds_left: 3600, idle_minutes: 60, extend_minutes: 5 }));
    if (path === "/api/auth/logout" || path === "/api/auth/identity") return Promise.resolve(reply({ ok: true }));
    if (path === "/api/auth/avatars") return Promise.resolve(reply(F.avatars));

    if (path === "/api/boards") return Promise.resolve(reply(F.boards));
    if (path === "/api/config") return Promise.resolve(reply(Object.assign({ ok: true }, F.config)));
    if (path === "/api/profiles") return Promise.resolve(reply(F.profiles));
    if (path.indexOf("/api/profiles/") === 0) return Promise.resolve(reply({ ok: true }));
    if (path === "/api/rag/docs") return Promise.resolve(reply(F.docs));
    if (path.indexOf("/api/rag/docs/") === 0) return Promise.resolve(reply({ ok: true }));
    if (path === "/api/logs") return Promise.resolve(reply(F.logs));
    if (path === "/api/reset") return Promise.resolve(reply({ ok: true }));

    if (path === "/api/resume") return Promise.resolve(reply(F.resume_upload));
    if (path === "/api/resume/detect")
      return Promise.resolve(reply({ ok: true, github_url: "https://github.com/arjunmehta", scanned: "default" }));
    if (path === "/api/resume/gen") return Promise.resolve(reply(F.resume_gen));
    if (path === "/api/resume/desc") return Promise.resolve(reply({ status: "empty", chars: 0 }));

    if (path === "/api/run" && method === "POST") return Promise.resolve(replay(F.run));

    if (path === "/api/gmail/sync")
      return Promise.resolve(reply({ ok: false, error: "Gmail sync needs the local app — the demo ships sample results." }));
    if (path.indexOf("/api/verify-board") === 0)
      return Promise.resolve(reply({ ok: false, error: "Manual wall solving needs the desktop app." }));
    if (path.indexOf("/api/candidate/") === 0 || path.indexOf("/api/recruiter/") === 0)
      return Promise.resolve(reply({ ok: false, error: NOT_HERE }));

    return Promise.resolve(reply({ ok: true }));
  };

  /* A demo that looks like the real thing should say it is not. */
  document.addEventListener("DOMContentLoaded", function () {
    var bar = document.createElement("div");
    bar.className = "demo-banner";
    bar.innerHTML = '<b>Static demo</b> · sample data from a real run · ' +
      '<a href="../index.html">about the project</a> · ' +
      '<a href="https://github.com/nitheeshkumarth-byte/JobScope" target="_blank" rel="noopener">source</a>';
    document.body.appendChild(bar);
  });
})();
