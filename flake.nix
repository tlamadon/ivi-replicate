{
  description = "PyTorch with CUDA on NixOS";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  inputs.scriptflow.url = "github:tlamadon/scriptflow?ref=feat%2Ftasks-wo-async";
  inputs.scriptflow.flake = false;  # <- important

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

      in {
        devShells.default = pkgs.mkShell {

          buildInputs = [
            pkgs.lazygit
            pkgs.uv
            pkgs.python312
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            echo "ml-earnings dev shell (use 'uv sync' to install dependencies)"
          '';
        };
      });
}
