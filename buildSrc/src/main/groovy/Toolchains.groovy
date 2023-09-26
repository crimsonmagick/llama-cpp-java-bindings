class Toolchains {

    static class ToolChain {

        ToolChain(final String file, final String generator) {
            this(file, generator, null)
        }

        ToolChain(final String file, final String generator, final Map<String, String> customDefs) {
            this.file = file
            this.generator = generator
            this.customDefs = customDefs ? customDefs : [:]
        }

        String file
        String generator
        Map<String, String> customDefs
    }

    private Toolchains() {}

    private static Map<String, ToolChain> toolChains = [:]

    static {
        toolChains[key(CompilerOs.LINUX, CompilerOs.LINUX, Architectures.x64, Compilers.GCC)] = new ToolChain('linux-x64-gcc-toolchain.cmake', 'Unix Makefiles')
        toolChains[key(CompilerOs.MACOS, CompilerOs.LINUX, Architectures.aarch64, Compilers.CLANG)] = new ToolChain('macos-linux-aarch64-clang-toolchain.cmake', 'Unix Makefiles')
        toolChains[key(CompilerOs.MACOS, CompilerOs.MACOS, Architectures.aarch64, Compilers.CLANG)] = new ToolChain('macos-aarch64-clang-toolchain.cmake', 'Unix Makefiles')
        toolChains[key(CompilerOs.WINDOWS, CompilerOs.LINUX, Architectures.x64, Compilers.MINGW)] = new ToolChain('windows-linux-mingw-toolchain.cmake', 'Unix Makefiles', ['JAVA_HOME': getWinJdkHome()])
        toolChains[key(CompilerOs.WINDOWS, CompilerOs.WINDOWS, Architectures.x64, Compilers.MINGW)] = new ToolChain('windows-mingw-toolchain.cmake', 'MinGW Makefiles')
        toolChains[key(CompilerOs.WINDOWS, CompilerOs.WINDOWS, Architectures.x64, Compilers.MSVC)] = new ToolChain('windows-msvc-toolchain.cmake', 'Visual Studio 17 2022')
    }

    static def getToolchain(final CompilerOs targetOs, CompilerOs hostOs, final Architectures architecture, final Compilers compiler) {
        return toolChains[key(targetOs, hostOs, architecture, compiler)]
    }

    private static def key(final CompilerOs targetOs, final CompilerOs hostOs, final Architectures architecture, Compilers compiler) {
        return "${targetOs.name()}-${hostOs.name()}-${architecture.name()}-${compiler.name()}}" as String
    }

    private static def getWinJdkHome() {
        final WIN_JAVA_HOME = 'WIN_JAVA_HOME'
        return System.hasProperty(WIN_JAVA_HOME) ? System.getProperty(WIN_JAVA_HOME) : System.getenv(WIN_JAVA_HOME) ? System.getenv(WIN_JAVA_HOME) : '/usr/local/include/jdk-17-win/jdk-17.0.8.1+1'
    }
}