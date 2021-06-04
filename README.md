# ML using Pytorch in C++

A simple repo to learn using Pytorch C++ API, for Deep Learning programs.

**Note**: This is just a repo, for quick references to setup `cmake`, use libtorch and other purposes. 
The code shouldn't be used in production, or anything serious.

## Installation and usage.

This project uses `CMake` of minimum `3.14` for dependency management. You need to ensure that you have `cmake` and `GCC runtime`
installed on your system.

Here's how to install the dependencies, and get started.

- Make a build folder, if it doesn't exist: `mkdir build`
- CD into the build folder: `cd build`
- Install the dependencies, using `cmake ..` and then `cmake --build . --config Release`

Here are the steps to run the executables, once compiled!

- Change to the respective directory for the program you wish to run.
  ```sh
  cd build/src/<the-folder-you-want-to-check>
  # Example: If I want to run the Regression example
  cd build/src/regression
  ```
- Run the executable for your Respective OS
  - Linux / MacOS
    ```sh
    ./<executable-file>
    # Example: If I want to run the Regression example
    ./regression  # Can change based on the name of executable
    ```
  - Windows
    ```powershell
    .\<executable-file>.exe
    # Example: If I want to run the Regression example
    .\regression.exe  # Can change based on the name of executable
    ```

## Contributing

Contributions, issues and feature requests are welcome. After cloning & setting up project locally, you
can just submit a PR to this repo and it will be deployed once it's accepted.

⚠️ It’s good to have descriptive commit messages, or PR titles so that other contributors can understand about your
commit or the PR Created. Read [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.3/)
before making the commit message.

To add a new section / topic, Get started by making a new folder, inside the `src/` folder.
Then, in the main `CMakeLists.txt`, Add `add_subdirectory("src/<project>")` with `<project>` being the 
project folder name. Add the project folder name to `add_dependencies(src` in the list.

Then, Make the files, and Add a configuration for it, in `CMakeLists.txt` in the specific folder you
made for project. Test if it works. Check the commands above and test and make it. And once it works, 
You're ready to contribute!

## Show your support

We love people's support in growing and improving. Be sure to leave a ⭐️ if you like the project and
also be sure to contribute, if you're interested!

<div align="center">
Made by Sunrit Jana with <3
</div>
