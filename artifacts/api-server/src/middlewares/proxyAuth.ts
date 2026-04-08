import { Request, Response, NextFunction } from "express";

export function proxyAuth(req: Request, res: Response, next: NextFunction): void {
  const proxyApiKey = process.env["PROXY_API_KEY"];
  if (!proxyApiKey) {
    res.status(500).json({ error: { message: "PROXY_API_KEY is not configured on the server.", type: "server_error" } });
    return;
  }

  const authHeader = req.headers["authorization"];
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    res.status(401).json({ error: { message: "Missing or malformed Authorization header. Expected: Bearer <key>", type: "invalid_request_error" } });
    return;
  }

  const providedKey = authHeader.slice("Bearer ".length).trim();
  if (providedKey !== proxyApiKey) {
    res.status(401).json({ error: { message: "Invalid API key.", type: "invalid_request_error" } });
    return;
  }

  next();
}
