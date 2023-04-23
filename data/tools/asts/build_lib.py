
from tree_sitter import Language

import subprocess
import os

# for lang in ['go', 'javascript', 'python', 'java', 'php', 'ruby', 'c-sharp']:
#     subprocess.run(['git', 'clone', f'git@github.com:tree-sitter/tree-sitter-{lang}.git',
#                     f'vendor/tree-sitter-{lang}'])

Language.build_library(
    # Store the library in the `build` directory
    '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/build/my-languages.so',

    # Include one or more languages
    [
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-go',
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-javascript',
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-python',
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-java',
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-php',
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-ruby',
        '/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/asts/vendor/tree-sitter-c-sharp'
    ]
)
