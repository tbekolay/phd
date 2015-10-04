from glob import glob
import os
import sys

import doit
from doit.action import CmdAction


def task_paper(root='.'):
    d = os.path.join(root, 'paper')

    def forsurecompile(fname, bibtex=True):
        pdf = CmdAction('pdflatex -interaction=nonstopmode %s.tex' % fname,
                        cwd=d)
        bib = CmdAction('bibtex %s' % fname, cwd=d)
        pdf_file = os.path.join(d, '%s.pdf' % fname)
        bib_file = os.path.join(d, '%s.bib' % fname)
        tex_files = glob(os.path.join(d, '*.tex'))
        return {'name': fname,
                'file_dep': tex_files + [bib_file] if bibtex else tex_files,
                'actions': [pdf, bib, pdf, pdf] if bibtex else [pdf, pdf],
                'targets': [pdf_file]}
    yield forsurecompile('phd')


def main():
    doit.run(globals())

if __name__ == '__main__':
    main()
