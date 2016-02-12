from glob import glob
import json
import os

import doit
from doit.action import CmdAction
import requests

from .tasks import *  # noqa; load experiment tasks

DOIT_CONFIG = {
    'default_tasks': [],
    'verbosity': 2,
}

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def task_download_data():
    """Downloads analyzed data from Figshare."""

    def download_data(article_id):
        r = requests.get("http://api.figshare.com/v1/articles/%s" % article_id)
        detail = json.loads(r.content)
        for file_info in detail['items'][0]['files']:
            outpath = os.path.join(root, file_info['name'])
            if os.path.exists(outpath):
                print("%s exists. Skipping." % outpath)
                continue
            with open(outpath, 'wb') as outf:
                print("Downloading %s..." % outpath)
                dl = requests.get(file_info['download_url'])
                outf.write(dl.content)

    cached = os.path.join(root, 'cache')
    n_files = sum([len(files) for r, d, files in os.walk(cached)])

    return {'actions': [(download_data, ['2064072']),
                        "unzip -o cache.zip",
                        "rm -f cache.zip"],
            'uptodate': [n_files >= 3562]}


def task_paper():
    """Compile thesis with LaTeX."""
    d = os.path.join(root, 'paper')

    def forsurecompile(fname, bibtex=True):
        pdf = CmdAction('texfot pdflatex -interaction=nonstopmode %s.tex' % fname,
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


def task_presentation():
    """Generate presentation slides."""
    figd = os.path.join(root, 'figures')
    presd = os.path.join(root, 'presentation')
    pfigd = os.path.join(presd, 'fig')
    reveald = os.path.join(presd, 'reveal.js')
    nbpath = os.path.join(presd, 'phd.ipynb')
    slpath = os.path.join(presd, 'index.html')

    def linkfigs():
        os.symlink(figd, pfigd)

    def dlreveal():
        with open("%s.zip" % reveald, 'w') as fp:
            dl = requests.get(
                "https://github.com/hakimel/reveal.js/archive/3.2.0.zip")
            fp.write(dl.content)

    yield {'name': 'linkfigs',
           'uptodate': [os.path.exists(pfigd)],
           'actions': [linkfigs]}

    yield {'name': 'dlreveal',
           'uptodate': [os.path.exists(reveald)],
           'actions': [dlreveal,
                       "unzip %s.zip -d %s" % (reveald, presd),
                       "mv %s-3.2.0 %s" % (reveald, reveald),
                       "rm -f %s.zip" % reveald]}

    yield {'name': 'generate',
           'file_dep': [nbpath, os.path.join(root, 'slides_reveal.tpl')],
           'targets': [slpath],
           'actions': ["jupyter nbconvert %s --to slides --output %s "
                       "--RevealHelpTransformer.url_prefix=reveal.js"
                       % (nbpath, nbpath),
                       "mv %s.slides.html %s" % (nbpath, slpath)]}

    yield {'name': 'serve',
           'actions': ["echo 'To serve the presentation, "
                       "cd presentation && python -m SimpleHTTPServer'"]}


def task_plots():
    """Run notebooks to generate plots."""
    figd = os.path.join(root, 'figures')
    modeld = os.path.join(root, 'models')

    for nb in os.listdir(modeld):
        if not nb.endswith('.ipynb'):
            continue

        nbin = os.path.join(modeld, nb)
        nbout = os.path.join(figd, nb)

        yield {'name': nb[:-6],
               'file_dep': [nbin],  # phd forget if other things change
               'actions': ['jupyter nbconvert --to notebook --execute %s '
                           '--ExecutePreprocessor.timeout=-1 '
                           '--output %s' % (nbin, nbout)],
               'targets': [nbout]}


def task_svg2pdf():
    """Convert SVGs to PDFs."""

    def svg2pdf(svgpath, pdfpath):
        return 'inkscape --export-pdf=%s %s' % (pdfpath, svgpath)

    d = os.path.join(root, 'figures')

    for fdir, _, fnames in os.walk(d):
        for fname in fnames:
            if fname.endswith('svg'):
                svgpath = os.path.join(root, fdir, fname)
                pdfpath = os.path.join(root, fdir, "%s.pdf" % fname[:-4])
                yield {'name': os.path.relpath(svgpath),
                       'actions': [svg2pdf(svgpath, pdfpath)],
                       'file_dep': [svgpath],
                       'targets': [pdfpath]}



def main():
    doit.run(globals())

if __name__ == '__main__':
    main()
