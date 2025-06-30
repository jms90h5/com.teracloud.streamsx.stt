# Investigation: Why SPL Compiler References Python venv

## The Mystery

The SPL compiler should only be dealing with SPL files and Streams resources, not Python virtual environments. So why is it trying to include `impl/venv/lib/python3.11/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt` in the build?

## Understanding SPL Application Bundle (.sab) Creation

When the SPL compiler creates a Streams Application Bundle (.sab file), it:

1. **Compiles SPL code** into C++ 
2. **Builds binaries** from the generated C++
3. **Packages everything** into a .sab file including:
   - Compiled binaries
   - Application descriptor (ADL)
   - **Toolkit dependencies**
   - **Data files and resources**

## The Root Cause: Toolkit Bundling

The issue appears to be in step 3 - when packaging toolkit dependencies. The SPL compiler seems to be:

1. **Scanning the entire toolkit directory** (despite .toolkit-ignore)
2. **Creating a file manifest** for the .sab bundle
3. **Generating makefile dependencies** for all files found

This happens because:
- The application references the toolkit with `-t ..`
- The compiler needs to bundle relevant toolkit files with the application
- The bundling process appears to ignore .toolkit-ignore in some cases

## Why Files with Spaces Break It

The generated Makefile has rules like:
```makefile
SimpleTest.sab: file1 file2 file3 ... Lorem ipsum.txt
```

Make interprets this as depending on two files: "Lorem" and "ipsum.txt"

## Evidence of Overly Broad Scanning

The fact that it's finding files deep in `impl/venv/lib/python3.11/site-packages/` suggests the scanner is:
- Recursively scanning ALL subdirectories
- Not properly honoring .toolkit-ignore
- Including non-SPL resources in the dependency list

## Potential SPL Compiler Issues

1. **Bundle Creation Logic**: The .sab bundling may be including too much
2. **.toolkit-ignore Processing**: May not be applied during dependency generation
3. **Makefile Generation**: Doesn't escape filenames with spaces
4. **Resource Detection**: May be too aggressive in what it considers a "resource"

## Why This Is a Bug

This behavior suggests several bugs in the SPL compiler:

1. **Over-inclusive bundling** - Python files should never be part of an SPL application
2. **Ignoring .toolkit-ignore** - The exclude file isn't being honored
3. **Poor filename handling** - Spaces in filenames break makefile generation
4. **No file type filtering** - Should only bundle relevant file types

## Workarounds Explained

Our workarounds address different aspects:

1. **Delete files with spaces** - Prevents makefile syntax errors
2. **Move venv outside** - Removes files from scanner's reach  
3. **Use --no-toolkit-indexing** - May skip some scanning (but breaks toolkit references)

## The Real Solution

The SPL compiler needs fixes:

1. **Honor .toolkit-ignore during ALL phases** including dependency scanning
2. **Filter file types** - Only include .spl, .xml, .so, data files
3. **Properly escape filenames** in generated makefiles
4. **Provide explicit exclude patterns** like `--exclude-pattern "*.py"`

## Filed Bug Report Template

```
Title: SPL Compiler includes Python venv files in application bundle dependencies

Description:
The SPL compiler (sc) incorrectly scans and includes Python virtual environment 
files when building applications that reference a toolkit containing a venv.

Steps to Reproduce:
1. Create toolkit with Python venv in impl/venv
2. Install Python packages (setuptools includes "Lorem ipsum.txt")  
3. Add "impl/venv/" to .toolkit-ignore
4. Compile SPL application with -t <toolkit>

Expected: Python files ignored, only SPL resources included
Actual: Build fails with "No rule to make target .../Lorem"

Root Causes:
- .toolkit-ignore not honored during dependency scanning
- All files included in makefile dependencies regardless of type
- Filenames with spaces not escaped in generated makefiles
```