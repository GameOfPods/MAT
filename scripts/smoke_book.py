"""
Quick end to end check of the book pipeline on a small generated EPUB.

    uv run python scripts/smoke_book.py
"""
import argparse
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

CHAPTERS = {
    "Chapter 1": ["Alice met Bob in Berlin on Monday.", "They both worked for Acme Corporation."],
    "Chapter 2": ["On Friday Bob flew to Paris.", "Alice stayed at home and read."],
    "Epilogue": ["In March 2024 they met again in London."],
}


def write_epub(path: Path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("mat-smoke")
    book.set_title("Smoke Book")
    book.set_language("en")
    items = []
    for i, (heading, paragraphs) in enumerate(CHAPTERS.items()):
        chapter = epub.EpubHtml(title=heading, file_name=f"chap_{i}.xhtml", lang="en")
        chapter.content = f"<html><body><h1>{heading}</h1>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"
        book.add_item(chapter)
        items.append(chapter)
    book.toc = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + items
    epub.write_epub(str(path), book)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model, default: %(default)s")
    parser.add_argument("--out", type=Path, default=None, help="Output folder, default: a temp folder")
    args = parser.parse_args()

    from MAT.pipelines import Pipeline
    from MAT.pipelines.Book import BookPipeline
    from MAT.reader import MATResult
    from MAT.utils.config import Config
    from MAT.writer import Writer

    out = args.out or Path(tempfile.mkdtemp(prefix="mat-smoke-"))
    out.mkdir(parents=True, exist_ok=True)
    book_path = out / "smoke.epub"
    write_epub(book_path)
    print(f"book: {book_path}")

    assert BookPipeline in Pipeline.get_pipelines(f=str(book_path)), "BookPipeline did not accept the epub"

    config = Config({"spacy": {"model": args.spacy_model}}, work_directory=str(out))
    config.validate()

    start = perf_counter()
    result = BookPipeline().process(file=str(book_path), config=config)
    print(f"took {perf_counter() - start:.1f}s")
    for chapter in result.chapter_data:
        print(chapter.get_beautiful_heading(), chapter.sentences, chapter.ner)

    assert [c.heading for c in result.chapter_data] == list(CHAPTERS.keys())
    assert all(c.ner for c in result.chapter_data), "NER did not run"

    folder = Writer().store(file=str(book_path), output=str(out / "results"), pipeline_results=[result])
    zipped = shutil.make_archive(folder, "zip", folder)
    for path in (folder, zipped):
        book = MATResult.read(path).book
        assert [c.heading for c in book.chapters] == list(CHAPTERS.keys()), f"could not read back {path}"
        assert any(s.entities for c in book.chapters for s in c.sentences), f"entities missing in {path}"
    print("SMOKE BOOK OK")


if __name__ == "__main__":
    main()
