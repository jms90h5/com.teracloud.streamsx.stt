# SPL Compiler Bug Report: Inappropriate Inclusion of Non-SPL Files in Build Dependencies

## Executive Summary

The SPL compiler (`sc`) incorrectly includes all files in a toolkit directory as build dependencies when creating Streams Application Bundles (.sab files), regardless of file type or relevance to SPL. This causes build failures when toolkit directories contain files with spaces in their names, such as those found in Python virtual environments.

## Bug Description

### Expected Behavior
When building an SPL application that references a toolkit (`-t <toolkit_path>`), the compiler should:
1. Only track SPL-relevant files (.spl, .xml, .so, declared data files)
2. Honor `.toolkit-ignore` exclusion patterns during all compilation phases
3. Properly escape filenames in generated makefiles

### Actual Behavior
The SPL compiler:
1. Recursively scans ALL files in the toolkit directory tree
2. Adds every file as a dependency in the generated makefile
3. Ignores `.toolkit-ignore` during dependency generation
4. Fails to escape filenames with spaces, causing makefile syntax errors

## Impact

This bug prevents toolkit developers from:
- Including Python virtual environments in their toolkit structure
- Having any files with spaces in filenames anywhere in the toolkit
- Organizing toolkits with non-SPL support files (documentation, tests, scripts)

## Reproduction Steps

1. Create a toolkit with standard structure:
```
my.toolkit/
├── info.xml
├── toolkit.xml
├── .toolkit-ignore
├── impl/
│   ├── venv/           # Python virtual environment
│   ├── src/
│   └── lib/
└── my.namespace/
    └── MyOperator/
```

2. Add Python virtual environment:
```bash
cd my.toolkit
python3 -m venv impl/venv
source impl/venv/bin/activate
pip install setuptools  # Creates "Lorem ipsum.txt" test file
```

3. Add exclusion to `.toolkit-ignore`:
```
impl/venv/
```

4. Build SPL application:
```bash
cd samples
sc -a -t ../my.toolkit -M MyApp MyApp.spl
```

5. **Result**: Build fails with:
```
make: *** No rule to make target '.../setuptools/_vendor/jaraco/text/Lorem', needed by 'MyApp.sab'.
```

## Root Cause Analysis

### File Discovery
The compiler scans the entire toolkit directory recursively, including:
- `impl/venv/lib/python3.11/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt`
- All other Python package files
- Any documentation, test files, or other non-SPL resources

### Makefile Generation
The generated makefile includes a rule like:
```makefile
MyApp.sab: file1.spl file2.xml ... Lorem ipsum.txt ...
```

Make interprets "Lorem ipsum.txt" as two separate targets: "Lorem" and "ipsum.txt"

### Evidence
1. Moving venv to different locations within the toolkit still triggers the bug
2. `.toolkit-ignore` is present but not honored during this phase
3. The error specifically mentions .sab file dependencies

## Severity: High

- **Blocks Development**: Forces non-standard toolkit organization
- **No Reasonable Workaround**: Requires manually deleting files or complex build processes
- **Affects Modern Development**: Python integration is common in analytics toolkits

## Proposed Solutions

### Short Term (Bug Fix)
1. **Honor .toolkit-ignore**: Apply exclusion patterns during dependency scanning
2. **Escape Filenames**: Properly quote filenames with spaces in makefiles
3. **Add --exclude flag**: Allow explicit exclusion patterns in sc command

### Long Term (Design Improvement)
1. **Smart Dependency Tracking**: Only track files that are actually referenced:
   - SPL files imported/used
   - Operator model XML files
   - Libraries referenced in operator models
   - Data files explicitly included
   
2. **File Type Filtering**: By default, only include:
   - `*.spl`
   - `*.xml` (operator models)
   - `*.so`, `*.a` (libraries)
   - Files in `data/` or `etc/` directories

3. **Explicit Resource Declaration**: Allow toolkit to declare which files should be bundled:
```xml
<!-- toolkit.xml -->
<toolkit>
  <resources>
    <include>data/**/*</include>
    <include>etc/config.properties</include>
    <exclude>**/*.pyc</exclude>
    <exclude>impl/venv/**</exclude>
  </resources>
</toolkit>
```

## Test Cases

1. **Spaces in Filenames**: Toolkit with "test file.txt" should compile
2. **Python venv**: Toolkit with impl/venv/ in .toolkit-ignore should compile
3. **Large Toolkits**: Performance should not degrade with many excluded files
4. **Special Characters**: Files with quotes, parentheses, etc. should work

## Workaround Documentation

Until fixed, document these workarounds:

1. **Remove problematic files**:
```bash
find toolkit_dir -name "* *" -type f -delete
```

2. **Use external Python environment**:
```bash
python3 -m venv ~/toolkit_venv
source ~/toolkit_venv/bin/activate
```

3. **Build in clean directory**:
```bash
rsync -av --exclude="venv" --exclude="*.pyc" toolkit/ /tmp/clean_toolkit/
sc -t /tmp/clean_toolkit ...
```

## Customer Impact Statement

This bug significantly impacts toolkit developers who need to:
- Include Python components for data science/ML operators
- Maintain comprehensive documentation within toolkits
- Follow modern development practices with virtual environments
- Share toolkits that "just work" without manual file cleanup

The current behavior forces developers to choose between:
- Excluding helpful development tools from their toolkit
- Requiring users to perform manual cleanup steps
- Using non-standard directory structures

## Recommendation

Implement the short-term fixes immediately to unblock developers, then design and implement the long-term improvements for the next major release. The current behavior appears to be a legacy design that doesn't align with modern toolkit development practices.