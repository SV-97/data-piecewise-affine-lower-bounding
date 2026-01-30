To set up CPLEX:

1. Create academic account with IBM at https://academic.ibm.com/a2mt/email-auth#/
2. Head to their Downloads thingy.
3. Try to download CPLEX. This causes a small bar to appear at the bottom of the page prompting you to install their `Download Director`.
4. Install their `Download director` (requires Java with WebStart support which appears to be somewhat finicky on linux. If this doesn't from your machine work try it from windows instead. You can still download the binaries for other platforms from there).
5. You can now download CPLEX from IBM's website.
6. Run the downloaded binary to install CPLEX optimization studio, noting down the installation path.
7. For python: install the `cplex` and `docplex` packages. Then run `docplex config --upgrade FULL_PATH_TO_YOUR_CPLEX_STUDIO_INSTALLATION` to make the python APIs use the installed (full) version rather than the community version that's bundled with the python packages.

