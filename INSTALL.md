
# Building `tiny-cuda-nn` from Source

## Prerequisites

- **CUDA Toolkit**: Make sure you have the CUDA Toolkit installed (version 11.8 in this case). Verify that `nvcc` is accessible in your PATH:
  ```bash
  /usr/local/cuda-11.8/bin/nvcc --version
  ```
  
- **C++ Compiler**: Ensure you have a compatible C++ compiler installed. GCC version 11.3.0 or later is recommended. Verify its presence:
  ```bash
  g++ --version
  ```

- **Ninja Build System**: Install Ninja to speed up the build process:
  ```bash
  pip install ninja
  ```

- **Additional Dependencies**: Install `cmake` and `setuptools`:
  ```bash
  pip install cmake setuptools
  ```

## Building from Source

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NVlabs/tiny-cuda-nn.git
   cd tiny-cuda-nn
   ```

2. **Create a Build Directory**:
   ```bash
   mkdir build
   cd build
   ```

3. **Run CMake**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=86
   ```

4. **Build the Project**:
   ```bash
   cmake --build .
   ```

## Common Issues and Solutions

### Issue: `gcc: fatal error: cannot execute ‘cc1plus’: execvp: No such file or directory`

**Solution**:
- Ensure that `g++` and `gcc` are correctly installed and accessible in your PATH. Update your `~/.bashrc` with the paths:
  ```bash
  export PATH=/usr/bin:$PATH
  ```

- Re-source your `~/.bashrc`:
  ```bash
  source ~/.bashrc
  ```

### Issue: `CUDA_ARCHITECTURES is empty for target`

**Solution**:
- Ensure that the CUDA Toolkit is correctly installed and `nvcc` is accessible. Verify installation:
  ```bash
  /usr/local/cuda-11.8/bin/nvcc --version
  ```

- Check that your `CMakeLists.txt` correctly specifies the CUDA architectures. You might need to adjust the `-DCMAKE_CUDA_ARCHITECTURES` flag.

### Issue: `nvcc fatal: Failed to preprocess host compiler properties`

**Solution**:
- Make sure that the correct version of GCC is used for CUDA compilation. The CUDA compiler requires a specific version of GCC that matches the version it was built with. Check compatibility and adjust GCC version if necessary.

## Useful Resources

- [tiny-cuda-nn GitHub Repository](https://github.com/NVlabs/tiny-cuda-nn)
- [Tiny CUDA NN Build Issue #183](https://github.com/NVlabs/tiny-cuda-nn/issues/183) - Contains useful information for troubleshooting build issues.
