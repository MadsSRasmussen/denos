type RunOptions = {
    cwd: string;
    capture: boolean;
};

export async function run(
    command: string[],
    options: Partial<RunOptions> = {},
) {
    const exec = new Deno.Command(command[0], {
        args: command.slice(1),
        cwd: options.cwd,
        stdout: options.capture ? "piped" : "inherit",
        stderr: "inherit",
    });
    const output = await exec.output();
    if (output.code !== 0) {
        error(
            `command "${
                command.join(" ")
            }" failed with exit code ${output.code}`,
        );
    }
    return options.capture
        ? new TextDecoder().decode(output.stdout).trim()
        : "";
}

export function error(message: string): never {
    console.error(message);
    Deno.exit(1);
}
