{
  description = "Moment Reconstruction of Image";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        customOverrides = self: super: {
          # Overrides go here
        };


        packageName = "MomentReconstruction";
      in {
#        packages.${packageName} = app;

#        defaultPackage = self.packages.${system}.${packageName};

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ 
              python3
              python3Packages.scikitimage
              python3Packages.scipy
              python3Packages.sympy
              pkgs.nodePackages.pyright
          ];
#          inputsFrom = builtins.attrValues self.packages.${system};
        };
      });
}

