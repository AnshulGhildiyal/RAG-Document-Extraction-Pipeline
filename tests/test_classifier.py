"""Tests for deterministic document classification."""

from rag_pipeline.core.classifier import DocumentClassifier


def test_classify_resume_text():
    classifier = DocumentClassifier()
    text = (
        "Curriculum Vitae\n"
        "Education: MCA\nExperience: Software Engineer\nSkills: Python, SQL, Machine Learning"
    )
    assert classifier.classify(text, filename="candidate_resume.pdf") == "resume"


def test_classify_invoice_text():
    classifier = DocumentClassifier()
    text = "Invoice #12345\nSubtotal: $100\nTax: $10\nTotal amount due: $110\nPayment due date: 2026-02-01"
    assert classifier.classify(text, filename="bill_jan.pdf") == "invoice"


def test_classify_contract_text():
    classifier = DocumentClassifier()
    text = (
        "This Agreement is made between Party A and Party B. "
        "The terms and conditions hereby state the obligations and effective date."
    )
    assert classifier.classify(text, filename="service_agreement.pdf") == "contract"


def test_classify_general_fallback():
    classifier = DocumentClassifier()
    text = "This is a short note with no strong domain keywords."
    assert classifier.classify(text, filename="notes.txt") == "general"
