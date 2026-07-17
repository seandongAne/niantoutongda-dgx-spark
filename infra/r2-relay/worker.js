// 回程通道 Worker:spark 无凭据分块上传的唯一入口。
// 鉴权 = 部署时注入的一次性随机 UPLOAD_TOKEN(每次传输重新部署换新);
// 传输完成后由 pull_results_r2.sh 用 disabled-token 重部署使其失效。
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return new Response("ok");
    }
    const match = url.pathname.match(/^\/up\/([^/]+)\/(.+)$/);
    if (!match) {
      return new Response("not found", { status: 404 });
    }
    const [, token, rawKey] = match;
    if (
      !env.UPLOAD_TOKEN ||
      env.UPLOAD_TOKEN.startsWith("disabled-") ||
      token !== env.UPLOAD_TOKEN
    ) {
      return new Response("forbidden", { status: 403 });
    }
    if (request.method !== "PUT") {
      return new Response("method not allowed", { status: 405 });
    }
    const key = decodeURIComponent(rawKey);
    // 只允许写本通道的会话前缀,防误覆盖桶内其他对象
    if (!key.startsWith("xfer-") || key.includes("..")) {
      return new Response("bad key", { status: 400 });
    }
    const object = await env.BUCKET.put(key, request.body);
    return Response.json({
      key: object.key,
      size: object.size,
      etag: object.httpEtag,
    });
  },
};
