{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310; # Define the Python version
  pythonPackages = python.pkgs; # Access to Python packages
in pkgs.mkShell rec {
  name = "python-ot-back-and-forth";
  venvDir = "./.venv"; # Specify the directory for the virtual environment

  shell = pkgs.fish;

  # Adding venvShellHook to manage the virtual environment
  buildInputs = [
    # jupyter lab needs
    pkgs.stdenv.cc.cc.lib

    pythonPackages.venvShellHook
    python
    pythonPackages.ipykernel
    pythonPackages.jupyterlab
    pythonPackages.notebook

    # System utilities
    pkgs.git
    pkgs.openssl
    pkgs.zlib
    pkgs.gcc

    # Additional libraries
    pkgs.libxml2
    pkgs.libxslt
    pkgs.libzip
  ];

  # Use the venvShellHook to automate the activation of the virtual environment
  # shellHook = ''
  #   echo "Activating virtual environment located in ${venvDir}..."
  # '';
  #
  # # Optional: Configuration for post-creation of virtual environment
  # postVenvCreation = [
  #   ''
  #     unset SOURCE_DATE_EPOCH
  #     # Optional: Install Jupyter kernel
  #     echo "Installing IPython kernel..."
  #     python -m ipykernel install --user --name=${name} --display-name="${name}"
  #   ''
  # ];
  #
  # postShellHook = ''
  #   unset SOURCE_DATE_EPOCH
  # '';

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    
    python -m ipykernel install --user --name=${name} --display-name="${name}"
    pip install -r requirements.txt
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
  '';


}
