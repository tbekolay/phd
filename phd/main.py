from glob import glob
import os

import doit
from doit.action import CmdAction

from .tasks import *  # noqa; Load experiment tasks

DOIT_CONFIG = {
    'default_tasks': [],
    'verbosity': 2,
}

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def task_paper():
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


def task_svg2pdf():

    def svg2pdf(svgpath, pdfpath):
        return 'inkscape --export-pdf=%s %s' % (pdfpath, svgpath)

    d = os.path.join(root, 'figures')

    for fdir, _, fnames in os.walk(d):
        for fname in fnames:
            if fname.endswith('svg'):
                svgpath = os.path.join(root, fdir, fname)
                pdfpath = os.path.join(root, fdir, "%s.pdf" % fname[:-4])
                yield {'name': os.path.basename(svgpath),
                       'actions': [svg2pdf(svgpath, pdfpath)],
                       'file_dep': [svgpath],
                       'targets': [pdfpath]}



def main():
    doit.run(globals())

if __name__ == '__main__':
    main()
