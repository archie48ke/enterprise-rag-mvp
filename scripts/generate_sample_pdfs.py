"""
Generate realistic sample PDF documents for all four domains so the RAG
pipeline can be demonstrated end-to-end without needing real company data.

Usage:
    python scripts/generate_sample_pdfs.py
"""

import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (current import name; `fitz` alias is deprecated)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def write_pdf(path: Path, pages_text: list):
    """Create a simple text PDF, one page per string in pages_text."""
    doc = fitz.open()
    for page_text in pages_text:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, page_text, fontsize=11, fontname="helv")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    print(f"Created: {path}")


HR_PAGES = [
    "HR Policy Handbook\nPage 1: Introduction\n\n"
    "This document describes the Human Resources policies applicable to all "
    "full-time employees of the company. It covers leave policy, working "
    "hours, and employee benefits.",

    "Leave Policy\nPage 2: Leave Types\n\n"
    "The company offers three categories of leave: casual leave, sick leave, "
    "and earned leave. Each leave type has its own approval workflow, "
    "described in the following sections.",

    "Leave Policy\nPage 3: Sick Leave\n\n"
    "Employees receive 8 sick leaves per calendar year. Sick leave can be "
    "taken with a maximum of 3 consecutive days without a medical certificate. "
    "Beyond 3 days, a medical certificate must be submitted to HR.",

    "Leave Policy\nPage 4: Casual Leave\n\n"
    "Employees receive 12 casual leaves per calendar year. Casual leave must "
    "be applied for at least 1 day in advance through the HR portal, except "
    "in emergencies. Unused casual leave does not carry over to the next year.",

    "Leave Policy\nPage 5: Earned Leave\n\n"
    "Employees accrue 1.5 earned leave days per month, up to a maximum of 18 "
    "days per year. Earned leave can be carried forward up to a maximum of "
    "45 days and can be encashed upon resignation, subject to manager approval.",
]

TECHNICAL_PAGES = [
    "Deployment Manual\nPage 1: Overview\n\n"
    "This manual describes how to deploy internal applications maintained by "
    "the Engineering team, including Project Alpha, using the standard "
    "company deployment pipeline.",

    "Deployment Manual\nPage 2: Prerequisites\n\n"
    "Before deploying Project Alpha, ensure the following prerequisites are "
    "installed on the target machine: Docker, Python 3.11, and Git. The "
    "deployment machine must also have at least 4GB of RAM available.",

    "Deployment Manual\nPage 3: Environment Variables\n\n"
    "Project Alpha deployment requires the following production environment "
    "variables to be set: DATABASE_URL, SECRET_KEY, and REDIS_URL. These "
    "values must be provided through the secrets manager, never committed to "
    "source control.",

    "Deployment Manual\nPage 4: Deployment Steps\n\n"
    "To deploy Project Alpha: 1) Pull the latest release tag from the "
    "repository. 2) Build the Docker image using the provided Dockerfile. "
    "3) Run database migrations. 4) Start the container using the production "
    "docker-compose file. 5) Verify the health check endpoint returns status 200.",

    "Deployment Manual\nPage 5: Rollback Procedure\n\n"
    "If a deployment fails, roll back by redeploying the previous stable "
    "image tag and re-running the database migration rollback script. "
    "Notify the on-call engineer before performing a rollback in production.",
]

PROJECTS_PAGES = [
    "Project Alpha Overview\nPage 1: Introduction\n\n"
    "Project Alpha is an internal customer management platform built by the "
    "Engineering team to centralize customer data across sales, support, and "
    "billing systems.",

    "Project Alpha Overview\nPage 2: Goals\n\n"
    "The primary goals of Project Alpha are to reduce duplicate customer "
    "records, provide a single dashboard for account managers, and integrate "
    "with the company's billing and support ticketing systems.",

    "Project Alpha Overview\nPage 3: Requirements\n\n"
    "Project Alpha requires integration with the existing CRM database, a "
    "role-based access control system, and an audit log for all customer "
    "data changes. It must support at least 10,000 concurrent customer records.",

    "Project Alpha Overview\nPage 4: Current Status\n\n"
    "Project Alpha is currently in active development. The customer data "
    "model and authentication modules are complete. The billing integration "
    "module is in progress and is expected to be complete next quarter.",

    "Project Alpha Overview\nPage 5: Team\n\n"
    "Project Alpha is owned by the Platform Engineering team, with support "
    "from the Data team for the customer deduplication logic and the "
    "Security team for access control review.",
]

GENERAL_PAGES = [
    "Company Information\nPage 1: About the Company\n\n"
    "The company builds enterprise software products for mid-size "
    "businesses. It was founded to help organizations manage internal "
    "knowledge more effectively.",

    "Company Information\nPage 2: Working Hours\n\n"
    "The company working hours are 9:00 AM to 6:00 PM, Monday through "
    "Friday. Employees are expected to be available during core hours of "
    "11:00 AM to 4:00 PM for meetings and collaboration.",

    "Company Information\nPage 3: Office Locations\n\n"
    "The company has two office locations: the headquarters and a regional "
    "office. Employees may work from either location or remotely, subject to "
    "manager approval and team requirements.",

    "Company Information\nPage 4: Holidays\n\n"
    "The company observes 10 public holidays per year, in addition to the "
    "leave entitlements described in the HR Leave Policy. The holiday "
    "calendar is published at the start of each calendar year.",

    "Company Information\nPage 5: Contact\n\n"
    "For general inquiries, employees can contact the internal help desk "
    "through the company intranet. HR-specific questions should be directed "
    "to the HR portal, and IT issues to the IT service desk.",
]


def main():
    write_pdf(DATA_DIR / "hr" / "leave_policy.pdf", HR_PAGES)
    write_pdf(DATA_DIR / "technical" / "deployment_manual.pdf", TECHNICAL_PAGES)
    write_pdf(DATA_DIR / "projects" / "project_alpha.pdf", PROJECTS_PAGES)
    write_pdf(DATA_DIR / "general" / "company_information.pdf", GENERAL_PAGES)
    print("\nSample PDFs generated. Now run: python scripts/build_indexes.py")


if __name__ == "__main__":
    main()
