import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import { logger } from "./lib/logger";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true, limit: "20mb" }));

app.use("/api", router);
app.use("/", router);

if (process.env.NODE_ENV === "production") {
  const uiDir = path.resolve(process.cwd(), "artifacts/proxy-ui/dist/public");
  if (fs.existsSync(uiDir)) {
    app.use(express.static(uiDir));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(uiDir, "index.html"));
    });
    logger.info({ uiDir }, "Serving proxy-ui static files");
  } else {
    logger.warn({ uiDir }, "proxy-ui static files not found, skipping static serve");
  }
}

export default app;
