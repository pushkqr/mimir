import unittest

from ingestion.metadata import extract_document_metadata


class IngestionMetadataTests(unittest.TestCase):
    def test_extract_document_metadata_finds_title_year_authority_and_number(self):
        markdown = """
        # Notification for Digital Services
        Department of Administrative Reforms
        Document No.: 2024-07
        Issued by: Ministry of Finance
        This circular provides guidance for public service delivery.
        """

        metadata = extract_document_metadata(markdown, "/tmp/sample.pdf", fallback_year=2025)

        self.assertEqual(metadata["document_title"], "Notification for Digital Services")
        self.assertEqual(metadata["year"], 2024)
        self.assertEqual(metadata["doc_number"], "2024-07")
        self.assertEqual(metadata["issuing_authority"], "Ministry of Finance")
        self.assertEqual(metadata["document_category"], "Notification")

    def test_supersedes_matches_the_original_phrasing(self):
        markdown = "Document No.: HTE-2022/88/TE-4\nThis Government Resolution is issued in supersession of Government Resolution No. HTE-2019/14/TE-4, dated 14.03.2019."
        metadata = extract_document_metadata(markdown, "/tmp/gr.pdf", fallback_year=2022)
        self.assertEqual(metadata["supersedes"], "HTE-2019/14/TE-4")

    def test_supersedes_matches_partial_modification_phrasing(self):
        markdown = "Document No.: HTE-2022/88/TE-4\nThis order is issued in partial modification of Government Resolution No. HTE-2019/14/TE-4, dated 14.03.2019."
        metadata = extract_document_metadata(markdown, "/tmp/gr.pdf", fallback_year=2022)
        self.assertEqual(metadata["supersedes"], "HTE-2019/14/TE-4")

    def test_supersedes_matches_continuation_phrasing(self):
        markdown = "Document No.: HTE-2022/88/TE-4\nThis circular is issued in continuation of Government Resolution No. HTE-2020/07/TE-4, dated 01.01.2020."
        metadata = extract_document_metadata(markdown, "/tmp/gr.pdf", fallback_year=2022)
        self.assertEqual(metadata["supersedes"], "HTE-2020/07/TE-4")

    def test_supersedes_excludes_own_document_number(self):
        markdown = "Document No.: HTE-2019/14/TE-4\nThis order is issued in continuation of Government Resolution No. HTE-2019/14/TE-4, dated 14.03.2019."
        metadata = extract_document_metadata(markdown, "/tmp/gr.pdf", fallback_year=2022)
        self.assertIsNone(metadata["supersedes"])

    def test_references_captures_inline_citations_without_a_label(self):
        markdown = (
            "Document No.: HTE-2022/88/TE-4\n"
            "This Government Resolution is issued vide Government Resolution No. HTE-2018/09/TE-4, "
            "dated 05.05.2018, read with Government Resolution No. HTE-2017/22/TE-4, dated 02.02.2017."
        )
        metadata = extract_document_metadata(markdown, "/tmp/gr.pdf", fallback_year=2022)
        self.assertIsNotNone(metadata["references"])
        self.assertIn("HTE-2018/09/TE-4", metadata["references"])
        self.assertIn("HTE-2017/22/TE-4", metadata["references"])

    def test_references_still_captures_the_labelled_block(self):
        markdown = "Document No.: HTE-2022/88/TE-4\nReference: Government Resolution No. HTE-2016/11/TE-4, dated 03.03.2016."
        metadata = extract_document_metadata(markdown, "/tmp/gr.pdf", fallback_year=2022)
        self.assertIn("HTE-2016/11/TE-4", metadata["references"])
