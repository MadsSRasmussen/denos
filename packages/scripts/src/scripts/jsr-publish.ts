/**
 * CI script for publishing a package to JSR based on a git tag.
 *
 * Reads the `GITHUB_REF_NAME` environment variable (expected to be set by a
 * GitHub Actions workflow) and parses it as either:
 *
 * - `v<version>` — publishes the package at the workspace root, or
 * - `<pkg>@v<version>` — bumps and publishes the workspace member whose
 *   directory name matches `<pkg>`, as declared in the root `deno.json`'s
 *   `workspace` array.
 *
 * The script updates the `version` field in the relevant `deno.json`, runs
 * `deno publish --allow-dirty`, and, if the version file changed, commits
 * and pushes that change back to the repository using a `github-actions[bot]`
 * git identity.
 *
 * @example Tag format for the workspace root
 * ```
 * GITHUB_REF_NAME=v1.2.3
 * ```
 *
 * @example Tag format for a workspace member named "scripts"
 * ```
 * GITHUB_REF_NAME=scripts@v1.2.3
 * ```
 *
 * @module
 */
import { error, run } from "../utils.ts";

async function main() {
    const tag = Deno.env.get("GITHUB_REF_NAME");
    if (!tag) error("no GITHUB_REF_NAME environment variable found");

    const TagPattern = /^(?:(?<pkg>[^@]+)@)?v(?<version>.+)$/;
    const match = tag.match(TagPattern);
    if (!match?.groups) error("GITHUB_REF_NAME did not match expected format");

    const { pkg, version } = match.groups;

    const config = JSON.parse(await Deno.readTextFile("deno.json"));

    let path: string | undefined = ".";

    /* Bump package version and publish */
    if (pkg && version) {
        /* Bump package version */
        if (!Array.isArray(config.workspace)) {
            error("no workspace array in deno.json");
        }

        path = (config.workspace as string[]).find((ws) =>
            ws.split("/").at(-1) === pkg
        );
        if (!path) error("workspace not found");
    } else if (!version) {
        error("version not specified");
    }

    const pacakgeConfig = pkg
        ? JSON.parse(
            await Deno.readTextFile(`${path}/deno.json`),
        )
        : config;
    if (!pacakgeConfig.version) error("no version field in deno.json");

    pacakgeConfig.version = version;
    await Deno.writeTextFile(
        `${path}/deno.json`,
        JSON.stringify(pacakgeConfig, null, 4) + "\n",
    );

    /* Publish to jsr via `deno publish` */
    await run(["deno", "publish", "--allow-dirty"], { cwd: path });

    /* Set bot as git user */
    await run(["git", "config", "user.name", "github-actions[bot]"]);
    await run([
        "git",
        "config",
        "user.email",
        "github-actions[bot]@users.noreply.github.com",
    ]);

    /* Commit changes to git */
    await run(["git", "add", `${path}/deno.json`]);
    const status = await run(
        ["git", "status", "--porcelain", "--", `${path}/deno.json`],
        { capture: true },
    );

    if (status === "") return; // Nothing to commit

    await run(["git", "commit", "-m", `chore: release ${tag}`]);
    await run(["git", "push", "origin", "HEAD"]);
}

if (import.meta.main) {
    await main();
}
