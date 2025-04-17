In this project, build options like CPU, NVIDIA, and AMD are defined using CMake's option() mechanism and propagated into the source code via a generated Config.h file. This is done using configure_file() with a Config.h.in template.

While it's possible to pass options directly via -DCPU=ON and rely on compiler definitions like -DCPU, this approach is less maintainable, harder to trace, and doesn't scale well. Using Config.h provides a clean, centralized, and IDE-friendly way to manage build-time configuration:

All available options are clearly declared in CMakeLists.txt

Active options are visible in the generated Config.h

It’s easy to inspect, debug, and extend

Consumers of the library (via install/export) can reuse the same config

This approach also helps avoid fragile target_compile_definitions() chains and supports safe installation of headers without leaking internal compiler flags.

