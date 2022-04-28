{
 description = "Moment Reconstruction of Image";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?rev=7859b5e8f7671af4c276a98833af40ea45f38090";
    flake-utils.url = "github:numtide/flake-utils";
    mach-nix = {url = "github:DavHau/mach-nix";};
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix}:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        customOverrides = self: super: {
          # Overrides go here
        };


    python = let
        packageOverrides = self: super: {
          scipy = super.scipy.overridePythonAttrs(old: rec {
            version = "1.8.0";
            src =  super.fetchPypi {
              pname = "scipy";
              inherit version;
              sha256 = "MdTy1rckvJqY5Se1hJuKflib8epjDDOqVj7akSyf8L0=";
             };
          });
        };
             in pkgs.python3.override {inherit packageOverrides; self = python;};

    #mach-nix.pypiDataRev = "7a1b507b1d8e23330793425677eed9dcc0e72b94";
    #python = mach-nix.lib."${system}".mkPython{
    #    requirements = ''
    #        numpy
    #        einops
    #        scikit-image
    #    '';
    #    providers = {
    #    };
    #};
      #customPython = pkgs.python39.buildEnv.override {
      #  extraLibs = [ sparse ];
      #};

        packageName = "MomentReconstruction";
      in {
#        packages.${packageName} = app;

#        defaultPackage = self.packages.${system}.${packageName};
        #app = pkgs.poetry2nix.mkPoetryApplication {
        #  projectDir = ./.;
        #  overrides =
        #    [ pkgs.poetry2nix.defaultPoetryOverrides customOverrides ];
        #};

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ 
              (python.withPackages (ps: [ps.scipy] ))
              python3Packages.numpy
              #python3
              #python3Packages.scikitimage
              #python39Packages.scipy
              #python3Packages.sympy
              #python3Packages.numpy
              python39Packages.sparse
              python3Packages.einops
              python39Packages.matplotlib
              python39Packages.jedi-language-server
              pkgs.nodePackages.pyright
              pkgs.mypy
          ];
#          inputsFrom = builtins.attrValues self.packages.${system};
        };
      });
}

