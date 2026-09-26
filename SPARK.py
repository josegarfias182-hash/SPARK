import os
import csv
import json
from datetime import datetime
from urllib.parse import quote
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
import openai
import pandas as pd
import pathlib
from rdkit.Chem import QED
import re
import time

# =========================
# Configuration / Clients
# =========================

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # <-- rotate/delete any hard-coded key!

HTTP_TIMEOUT = 20  # seconds
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# PubChem request controls
PUBCHEM_REQUEST_DELAY = 1.0
PUBCHEM_MAX_RETRIES = 3

PUBCHEM_CACHE_FILE = "pubchem_cache.json"


def load_pubchem_cache():
    """Load previously retrieved PubChem data from disk."""
    if not os.path.exists(PUBCHEM_CACHE_FILE):
        return {}

    try:
        with open(PUBCHEM_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception as e:
        print(f"⚠️ Could not read PubChem cache: {e}")

    return {}

def save_pubchem_cache(cache):
    """Save successfully retrieved PubChem data to disk."""
    try:
        with open(PUBCHEM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    except Exception as e:
        print(f"⚠️ Could not save PubChem cache: {e}")


PUBCHEM_CACHE = load_pubchem_cache()

def pubchem_get(url, timeout=None, params=None):
    """
    Send a rate-limited GET request to PubChem.

    Temporary PubChem/server errors are retried using exponential
    backoff rather than being interpreted as missing chemical data.
    """

    if timeout is None:
        timeout = HTTP_TIMEOUT

    for attempt in range(PUBCHEM_MAX_RETRIES):

        try:
            # Keep automated requests below PubChem's request-rate guidance
            time.sleep(PUBCHEM_REQUEST_DELAY)

            response = requests.get(
                url,
                params=params,
                timeout=timeout
            )

            # Success
            if response.status_code == 200:
                return response

            # Temporary errors: retry
            if response.status_code in (429, 500, 502, 503, 504):

                wait_time = 2 ** attempt

                print(
                    f"⚠️ PubChem HTTP {response.status_code}. "
                    f"Retry {attempt + 1}/{PUBCHEM_MAX_RETRIES} "
                    f"in {wait_time} second(s)..."
                )

                time.sleep(wait_time)
                continue

            # A 404 generally means PubChem could not find that resource
            if response.status_code == 404:
                return response

            print(
                f"⚠️ PubChem request returned HTTP "
                f"{response.status_code}"
            )

            return response

        except requests.RequestException as e:

            wait_time = 2 ** attempt

            print(
                f"⚠️ PubChem network error: {e}. "
                f"Retry {attempt + 1}/{PUBCHEM_MAX_RETRIES} "
                f"in {wait_time} second(s)..."
            )

            time.sleep(wait_time)

    print(
        f"❌ PubChem request failed after "
        f"{PUBCHEM_MAX_RETRIES} attempts."
    )

    return None

# =========================
# GPT Helpers (single path)
# =========================
def chat_with_gpt(prompt: str, system: str = "You are an expert in medicinal chemistry.", model: str = "gpt-4o-mini") -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"GPT Error: {e}"

# =========================
# Descriptor Calculators
# =========================

from rdkit.Chem import rdMolDescriptors

def calculate_logS_esol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Invalid SMILES"
    logP = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heavy = Descriptors.HeavyAtomCount(mol)
    frac_aromatic = (aromatic_atoms / heavy) if heavy else 0.0
    logS = 0.16 - 0.63 * logP - 0.0062 * mw + 0.066 * rot - 0.74 * frac_aromatic
    return round(logS, 3)

def calculate_rdkit_descriptors_full(smiles: str):
    """Return all core physchem props needed for rules + formula & "Exact Mass"."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    # Make a canonical, explicit-H version if you like:
    can = Chem.MolToSmiles(mol, canonical=True)
    return {
        "SMILES": can,
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "Exact Mass": round(rdMolDescriptors.CalcExactMolWt(mol), 6),
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "LogS (ESOL)": calculate_logS_esol(smiles),
        "QED": round(QED.qed(mol), 3),
    }

def count_heteroatoms(smiles: str) -> int:
    """
    Count heteroatoms (non-C, non-H) in a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (1, 6):  # not H, not C
            count += 1
    return count

def classify_solubility(logS):
    """
    Classify solubility from ESOL LogS.
    """
    if isinstance(logS, (int, float)):
        if logS >= -2:
            return "Highly soluble"
        elif -4 <= logS < -2:
            return "Moderately soluble"
        elif -6 <= logS < -4:
            return "Poorly soluble"
        else:
            return "Very poorly soluble"
    return "Unknown"
def prepare_smiles_for_lookup(smiles: str) -> dict:
    """
    Validate an input SMILES with RDKit and generate standardized
    identifiers for database lookup.

    The original input SMILES remains the primary structure
    supplied to SPARK.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return {
                "valid": False,
                "original": smiles,
                "canonical": None,
                "inchikey": None,
            }

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )

        inchikey = Chem.MolToInchiKey(mol)

        return {
            "valid": True,
            "original": smiles,
            "canonical": canonical,
            "inchikey": inchikey,
        }

    except Exception:
        return {
            "valid": False,
            "original": smiles,
            "canonical": None,
            "inchikey": None,
        }
    
# =========================
# PubChem helpers
# =========================
def get_pubchem_cid(query: str, is_smiles: bool):
    """
    Resolve a compound to a PubChem CID.

    For SMILES input:
      1. Validate the supplied SMILES using RDKit.
      2. Generate an InChIKey from the supplied structure.
      3. Check the local PubChem cache.
      4. If not cached, query PubChem using the InChIKey.

    The supplied SMILES remains the primary structural input to SPARK.
    """

    if not query:
        return None, "No query"

    # =========================================================
    # SMILES INPUT
    # =========================================================

    if is_smiles:

        prepared = prepare_smiles_for_lookup(query)

        if not prepared["valid"]:
            print(f"⚠️ RDKit could not parse SMILES: {query}")
            return None, "Invalid SMILES"

        inchikey = prepared.get("inchikey")

        if not inchikey:
            print(
                "⚠️ RDKit parsed the SMILES but could not "
                "generate an InChIKey."
            )
            return None, "Identifier generation failed"

        print(
            f"🔎 PubChem lookup using RDKit InChIKey: "
            f"{inchikey}"
        )

        # ---------------------------------------------------------
        # Check local PubChem cache first
        # ---------------------------------------------------------

        cached = PUBCHEM_CACHE.get(inchikey)

        if isinstance(cached, dict) and cached.get("cid"):
            cached_cid = cached["cid"]

            print(
                f"💾 PubChem CID {cached_cid} loaded from local cache "
                f"for InChIKey {inchikey}."
            )

            return cached_cid, "Matched (cached)"

        # ---------------------------------------------------------
        # Not cached -> query PubChem
        # ---------------------------------------------------------

        try:
            url = (
                f"{PUBCHEM_BASE}/compound/inchikey/"
                f"{quote(inchikey, safe='')}/cids/JSON"
            )

            r = pubchem_get(url)

            if r is None:
                print(
                    "⚠️ PubChem was unavailable after retries "
                    f"for InChIKey {inchikey}."
                )
                return None, "PubChem unavailable"

            if r.status_code == 404:
                print(
                    "⚠️ PubChem did not contain a compound "
                    f"matching InChIKey {inchikey}."
                )
                return None, "Not found"

            if r.status_code != 200:
                print(
                    f"⚠️ PubChem lookup returned HTTP "
                    f"{r.status_code} for InChIKey {inchikey}."
                )
                return None, f"PubChem HTTP {r.status_code}"

            data = r.json()

            cids = (
                data.get("IdentifierList", {})
                .get("CID", [])
            )

            if cids:
                cid = cids[0]

                print(
                    f"✅ PubChem CID {cid} resolved "
                    f"using InChIKey {inchikey}."
                )

                # Save successful identity resolution
                if inchikey not in PUBCHEM_CACHE:
                    PUBCHEM_CACHE[inchikey] = {}

                PUBCHEM_CACHE[inchikey]["cid"] = cid

                save_pubchem_cache(PUBCHEM_CACHE)

                return cid, "Matched"

            print(
                "⚠️ PubChem returned a successful response, "
                "but no CID was present."
            )

            return None, "No CID returned"

        except Exception as e:
            print(
                f"⚠️ PubChem InChIKey lookup error: {e}"
            )
            return None, "PubChem lookup error"

    # =========================================================
    # NAME INPUT
    # =========================================================

    else:

        try:
            url = (
                f"{PUBCHEM_BASE}/compound/name/"
                f"{quote(query, safe='')}/cids/JSON"
            )

            r = pubchem_get(url)

            if r is None:
                return None, "PubChem unavailable"

            if r.status_code == 404:
                return None, "Not found"

            if r.status_code != 200:
                return None, f"PubChem HTTP {r.status_code}"

            data = r.json()

            cids = (
                data.get("IdentifierList", {})
                .get("CID", [])
            )

            if cids:
                cid = cids[0]

                print(
                    f"✅ PubChem CID {cid} resolved "
                    f"using compound name."
                )

                return cid, "Matched"

            return None, "No CID returned"

        except Exception as e:
            print(
                f"⚠️ PubChem name lookup error "
                f"for {query}: {e}"
            )

            return None, "PubChem lookup error"

def get_properties_by_cid(cid: int):
    """
    Retrieve core physicochemical properties from PubChem.

    PubChem values are treated as the primary externally sourced
    values. Successfully retrieved results are stored in the local
    cache so they do not need to be downloaded again.
    """

    if not cid:
        return {}

    # ---------------------------------------------------------
    # Check local cache for previously retrieved properties
    # ---------------------------------------------------------

    for inchikey, cached in PUBCHEM_CACHE.items():

        if (
            isinstance(cached, dict)
            and cached.get("cid") == cid
            and isinstance(cached.get("properties"), dict)
            and cached["properties"]
        ):
            print(
                f"💾 PubChem properties for CID {cid} "
                f"loaded from local cache."
            )

            return cached["properties"]

    # ---------------------------------------------------------
    # Not cached -> retrieve from PubChem
    # ---------------------------------------------------------

    try:
        props = (
            "Title,"
            "IUPACName,"
            "MolecularFormula,"
            "MolecularWeight,"
            "SMILES,"
            "ConnectivitySMILES,"
            "InChIKey,"
            "XLogP,"
            "ExactMass,"
            "TPSA,"
            "HBondDonorCount,"
            "HBondAcceptorCount,"
            "RotatableBondCount"
        )

        url = (
            f"{PUBCHEM_BASE}/compound/cid/{cid}/"
            f"property/{props}/JSON"
        )

        r = pubchem_get(url)

        if r is None:
            print(
                f"⚠️ PubChem property request failed after retries "
                f"for CID {cid}"
            )
            return {}

        if r.status_code != 200:
            print(
                f"⚠️ PubChem property request failed for CID {cid}: "
                f"HTTP {r.status_code}"
            )
            return {}

        data = r.json()

        records = (
            data.get("PropertyTable", {})
            .get("Properties", [])
        )

        if not records:
            print(
                f"⚠️ No PubChem property record returned "
                f"for CID {cid}"
            )
            return {}

        rec = records[0]

        out = {}

        # --------------------------------------------------
        # PubChem identity
        # --------------------------------------------------

        if rec.get("Title"):
            out["PubChem Title"] = rec["Title"]

        if rec.get("IUPACName"):
            out["IUPAC Name"] = rec["IUPACName"]

        if rec.get("InChIKey"):
            out["InChIKey"] = rec["InChIKey"]

        # --------------------------------------------------
        # PubChem structure
        # --------------------------------------------------

        if rec.get("SMILES"):
            out["SMILES"] = rec["SMILES"]

        elif rec.get("ConnectivitySMILES"):
            out["SMILES"] = rec["ConnectivitySMILES"]

        if rec.get("ConnectivitySMILES"):
            out["PubChem Connectivity SMILES"] = (
                rec["ConnectivitySMILES"]
            )

        # --------------------------------------------------
        # PubChem physicochemical properties
        # --------------------------------------------------

        if rec.get("MolecularFormula") is not None:
            out["Molecular Formula"] = rec["MolecularFormula"]

        if rec.get("MolecularWeight") is not None:
            out["Molecular Weight"] = rec["MolecularWeight"]

        if rec.get("ExactMass") is not None:
            out["Exact Mass"] = rec["ExactMass"]

        if rec.get("XLogP") is not None:
            out["LogP"] = rec["XLogP"]

        if rec.get("TPSA") is not None:
            out["TPSA"] = rec["TPSA"]

        if rec.get("HBondAcceptorCount") is not None:
            out["HBA"] = rec["HBondAcceptorCount"]

        if rec.get("HBondDonorCount") is not None:
            out["HBD"] = rec["HBondDonorCount"]

        if rec.get("RotatableBondCount") is not None:
            out["Rotatable Bonds"] = rec["RotatableBondCount"]

        # --------------------------------------------------
        # Save successful PubChem result to cache
        # --------------------------------------------------

        record_inchikey = out.get("InChIKey")

        if record_inchikey:

            if record_inchikey not in PUBCHEM_CACHE:
                PUBCHEM_CACHE[record_inchikey] = {}

            PUBCHEM_CACHE[record_inchikey]["cid"] = cid
            PUBCHEM_CACHE[record_inchikey]["properties"] = out

            save_pubchem_cache(PUBCHEM_CACHE)

            print(
                f"💾 PubChem properties for CID {cid} "
                f"saved to local cache."
            )

        return out

    except requests.RequestException as e:
        print(
            f"⚠️ PubChem network error for CID {cid}: {e}"
        )
        return {}

    except Exception as e:
        print(
            f"⚠️ Error parsing PubChem properties "
            f"for CID {cid}: {e}"
        )
        return {}


def get_detailed_properties(cid: int):
    """
    Retrieve detailed PubChem properties from a single PUG View request.

    Extracts:
      - Melting Point
      - Boiling Point
      - pKa

    Molecular Formula is already retrieved by get_properties_by_cid(),
    so it is not requested/extracted again here.
    """

    out = {
        "Melting Point": "Not found",
        "Boiling Point": "Not found",
        "pKa": None,
    }

    if not cid:
        return out

    try:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/"
            f"pug_view/data/compound/{cid}/JSON"
        )

        r = pubchem_get(url)

        if r is None:
            return out

        if r.status_code != 200:
            return out

        data = r.json()
        sections = data.get("Record", {}).get("Section", []) or []

        # ---------------------------------------------------------
        # Extract a value from a PubChem section by TOC heading
        # ---------------------------------------------------------

        def extract_by_heading(secs, label):
            for section in secs:

                heading = section.get("TOCHeading", "") or ""

                if heading == label:

                    # Information may be directly inside the section
                    for info in section.get("Information", []):

                        values = (
                            info.get("Value", {})
                            .get("StringWithMarkup", [])
                        )

                        for item in values:
                            value = item.get("String")

                            if value:
                                return value

                    # Information may also be inside subsections
                    for subsection in section.get("Section", []):

                        for info in subsection.get("Information", []):

                            values = (
                                info.get("Value", {})
                                .get("StringWithMarkup", [])
                            )

                            for item in values:
                                value = item.get("String")

                                if value:
                                    return value

                # Recursively search nested sections
                nested = section.get("Section", [])

                if nested:
                    value = extract_by_heading(nested, label)

                    if value:
                        return value

            return None

        # ---------------------------------------------------------
        # Extract pKa
        # ---------------------------------------------------------

        def extract_pka(secs):
            for section in secs:

                heading = section.get("TOCHeading", "") or ""

                if "pka" in heading.lower():

                    for info in section.get("Information", []):

                        values = (
                            info.get("Value", {})
                            .get("StringWithMarkup", [])
                        )

                        for item in values:
                            value = item.get("String")

                            if value:
                                return value

                    # pKa information can also be nested
                    for subsection in section.get("Section", []):

                        for info in subsection.get("Information", []):

                            values = (
                                info.get("Value", {})
                                .get("StringWithMarkup", [])
                            )

                            for item in values:
                                value = item.get("String")

                                if value:
                                    return value

                nested = section.get("Section", [])

                if nested:
                    value = extract_pka(nested)

                    if value:
                        return value

            return None

        melting_point = extract_by_heading(
            sections,
            "Melting Point"
        )

        boiling_point = extract_by_heading(
            sections,
            "Boiling Point"
        )

        pka_value = extract_pka(sections)

        if melting_point:
            out["Melting Point"] = melting_point

        if boiling_point:
            out["Boiling Point"] = boiling_point

        if pka_value:
            out["pKa"] = pka_value

        return out

    except Exception as e:
        print(
            f"⚠️ Error parsing PubChem detailed properties "
            f"for CID {cid}: {e}"
        )

        return out
    
def get_pubchem_names_and_synonyms(cid: int, core_properties=None) -> dict:
    """
    Return PubChem title, IUPAC name, InChIKey, and synonyms.

    Title, IUPAC name, and InChIKey are reused from the core
    PubChem property request when available. This avoids making
    a duplicate PubChem property request.

    Only the synonym endpoint requires an additional request.
    """

    out = {
        "title": None,
        "iupac": None,
        "inchikey": None,
        "synonyms": []
    }

    if not cid:
        return out

    # ---------------------------------------------------------
    # Reuse data already retrieved by get_properties_by_cid()
    # ---------------------------------------------------------

    if isinstance(core_properties, dict):
        out["title"] = core_properties.get("PubChem Title")
        out["iupac"] = core_properties.get("IUPAC Name")
        out["inchikey"] = core_properties.get("InChIKey")

    # ---------------------------------------------------------
    # Retrieve synonyms only
    # ---------------------------------------------------------

    try:
        url = (
            f"{PUBCHEM_BASE}/compound/cid/{cid}/"
            f"synonyms/JSON"
        )

        r = pubchem_get(url)

        if r is not None and r.status_code == 200:

            information = (
                r.json()
                .get("InformationList", {})
                .get("Information", [])
            )

            if information:
                syns = information[0].get("Synonym", []) or []

                cleaned = []
                seen = set()

                for synonym in syns:
                    synonym = synonym.strip()

                    if (
                        synonym
                        and synonym not in seen
                        and len(synonym) <= 60
                    ):
                        cleaned.append(synonym)
                        seen.add(synonym)

                out["synonyms"] = cleaned[:20]

    except Exception as e:
        print(
            f"⚠️ PubChem synonym lookup error "
            f"for CID {cid}: {e}"
        )

    return out

# =========================
# ChEMBL helpers
# =========================
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_TIMEOUT = 15

def looks_like_chembl_id(s: str) -> bool:
    s = s.strip().upper()
    return s.startswith("CHEMBL") and s[6:].isdigit()

def chembl_get_molecule(chembl_id: str) -> dict:
    """Fetch a single ChEMBL molecule record by CHEMBL ID."""
    url = f"{CHEMBL_API}/molecule/{chembl_id}.json"

    try:
        r = requests.get(url, timeout=CHEMBL_TIMEOUT)

        if r.status_code != 200:
            return {}

        return r.json() or {}

    except Exception:
        return {}

def chembl_search_molecule(query: str) -> dict:
    """
    Try searches to resolve free text or SMILES to a ChEMBL molecule.

    Strategy:
      - direct molecule lookup if query is already a CHEMBL ID
      - otherwise use ChEMBL text search

    Returns:
        Full molecule dictionary for the first hit, or {}.
    """

    q = query.strip()

    if looks_like_chembl_id(q):
        rec = chembl_get_molecule(q.upper())
        return rec or {}

    try:
        url = (
            f"{CHEMBL_API}/molecule/search.json"
            f"?q={quote(q)}"
        )

        r = requests.get(
            url,
            timeout=CHEMBL_TIMEOUT
        )

        if r.status_code != 200:
            return {}

        data = r.json() or {}
        hits = data.get("molecules") or []

        if hits:
            chembl_id = hits[0].get("molecule_chembl_id")

            if chembl_id:
                return chembl_get_molecule(chembl_id)

    except Exception:
        pass

    return {}

def chembl_extract_props(mol: dict) -> dict:
    """
    Extract a compact set of useful fields from a ChEMBL molecule record.

    ChEMBL may return ATC classifications as strings or dictionaries,
    so both formats are handled safely.
    """
    if not mol:
        return {}

    props = mol.get("molecule_properties") or {}
    structs = mol.get("molecule_structures") or {}
    atc = mol.get("atc_classifications") or []

    # -----------------------------------------
    # Safely extract ATC classification codes
    # -----------------------------------------
    atc_codes = []

    for item in atc:

        # Some ChEMBL records may return dictionaries
        if isinstance(item, dict):

            code = (
                item.get("level5")
                or item.get("level4")
                or item.get("level3")
                or item.get("level2")
                or item.get("level1")
            )

            if code:
                atc_codes.append(str(code).strip())

        # Other ChEMBL records may return the ATC code directly as a string
        elif isinstance(item, str):

            if item.strip():
                atc_codes.append(item.strip())

    # Remove duplicates
    atc_codes = sorted(set(atc_codes))

    out = {
        "ChEMBL ID": mol.get("molecule_chembl_id"),
        "ChEMBL Pref Name": mol.get("pref_name"),
        "ChEMBL SMILES": structs.get("canonical_smiles"),
        "ChEMBL InChIKey": structs.get("standard_inchi_key"),
        "ChEMBL QED": _to_float(props.get("qed_weighted")),
        "ChEMBL ALogP": _to_float(props.get("alogp")),
        "ChEMBL MW Freebase": _to_float(props.get("mw_freebase")),
        "ChEMBL PSA": _to_float(props.get("psa")),
        "ChEMBL HBA": _to_float(props.get("hba")),
        "ChEMBL HBD": _to_float(props.get("hbd")),
        "ChEMBL RO5 Violations": _to_float(props.get("ro5_violations")),
        "ChEMBL ATC Codes": ", ".join(atc_codes) if atc_codes else "Not available",
    }

    # Remove empty fields
    return {
        k: v
        for k, v in out.items()
        if v not in (None, "", [])
    }

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def chembl_bioactivity_summary(chembl_id: str) -> dict:
    """
    Pull a light summary of ChEMBL bioactivity counts
    and maximum pChEMBL value.
    """
    try:
        page_size = 200

        url = (
            f"{CHEMBL_API}/activity.json"
            f"?molecule_chembl_id={chembl_id}"
            f"&limit={page_size}"
        )

        r = requests.get(
            url,
            timeout=CHEMBL_TIMEOUT
        )

        if r.status_code != 200:
            return {}

        data = r.json() or {}
        acts = data.get("activities") or []

        total = 0
        max_pchembl = None

        for a in acts:
            total += 1

            p = a.get("pchembl_value")

            if p is not None:
                try:
                    p = float(p)

                    max_pchembl = (
                        p
                        if max_pchembl is None or p > max_pchembl
                        else max_pchembl
                    )

                except Exception:
                    pass

        return {
            "ChEMBL Bioactivity Count": total,
            "ChEMBL Max pChEMBL": (
                round(max_pchembl, 3)
                if max_pchembl is not None
                else "Not available"
            ),
        }

    except Exception:
        return {}

# =========================
# Utility
# =========================
def looks_like_smiles(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # very light heuristic
    return any(c in t for c in "=#[]()/\\") and not t.replace(" ", "").isalpha()

# =========================
# PubMed / NCBI E-utilities
# =========================
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL   = "medchem-assistant"
NCBI_EMAIL  = os.getenv("PUBMED_EMAIL", "jose.garfias182@gmail.com")   # set via env if possible
NCBI_APIKEY = os.getenv("NCBI_API_KEY")                      # optional, raises rate limits

def _http_get(url, params, timeout=20):
    params = {**params, "tool": NCBI_TOOL, "email": NCBI_EMAIL}
    if NCBI_APIKEY:
        params["api_key"] = NCBI_APIKEY
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r

def pubmed_esearch(query, retmax=15, sort="bestmatch", mindate=None, maxdate=None, filters=None):
    """
    filters: list of PubMed filters, e.g. ["review", "clinicaltrial", "humans"]
    """
    term = query
    if filters:
        f = " AND ".join(f'"{x}"[Filter]' for x in filters)
        term = f"({query}) AND ({f})"
    params = {"db": "pubmed", "term": term, "retmax": str(retmax), "retmode": "json", "sort": sort}
    if mindate or maxdate:
        params.update({"mindate": mindate or "", "maxdate": maxdate or "", "datetype": "pdat"})
    data = _http_get(f"{NCBI_EUTILS}/esearch.fcgi", params).json()
    return data.get("esearchresult", {}).get("idlist", []) or []

def pubmed_fetch_summaries(pmids):
    """Return minimal records (title, journal, year, doi)."""
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json", "rettype": "abstract"}
    js = _http_get(f"{NCBI_EUTILS}/esummary.fcgi", params).json()
    out, res = [], js.get("result", {})
    for uid in res.get("uids", []):
        s = res.get(uid, {})
        year = ""
        try:
            year = (s.get("pubdate") or "").split(" ")[0].split("-")[0]
        except Exception:
            pass
        doi = ""
        for iden in s.get("articleids", []) or []:
            if iden.get("idtype") == "doi":
                doi = iden.get("value", ""); break
        out.append({
            "pmid": uid,
            "title": (s.get("title") or "").strip(),
            "journal": (s.get("fulljournalname") or s.get("source") or "").strip(),
            "year": year,
            "doi": doi,
        })
    return out

def pubmed_fetch_abstracts(pmids):
    """Return dict pmid->abstract text (plain)."""
    if not pmids:
        return {}
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    xml = _http_get(f"{NCBI_EUTILS}/efetch.fcgi", params).text
    # Lightweight XML scrape for AbstractText blocks
    chunks = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, flags=re.S|re.I)
    clean = [re.sub("<[^>]+>", " ", c) for c in chunks]
    clean = [re.sub(r"\s+", " ", c).strip() for c in clean]
    # Map by order best-effort (good enough for top-N)
    return {pmid: clean[i] for i, pmid in enumerate(pmids) if i < len(clean)}

def build_generic_literature_query(matched_row: dict, user_q: str) -> str:
    """
    Compose a robust PubMed query from user question + compound context.
    Priority of names:
      PubChem Title > ChEMBL Pref Name > row['Compound'] (if it's not just a SMILES)
      + a few PubChem synonyms.
    """
    # 1) Gather human-readable names
    candidates: list[str] = []

    # PubChem and ChEMBL names
    for k in ["PubChem Title", "ChEMBL Pref Name"]:
        v = (matched_row.get(k) or "").strip()
        if v and v.lower() != "not found":
            candidates.append(v)

    # Original compound field, if it's not just a SMILES
    comp = (matched_row.get("Compound") or "").strip()
    if comp and not looks_like_smiles(comp):
        candidates.append(comp)

    # A few synonyms if we have them (PubChem)
    syns_str = (matched_row.get("Synonyms (PubChem)") or "").strip()
    if syns_str:
        for s in syns_str.split(";"):
            s2 = s.strip()
            if s2 and len(s2) >= 3:
                candidates.append(s2)

    # Deduplicate names while preserving order
    seen = set()
    names: list[str] = []
    for n in candidates:
        if n not in seen:
            names.append(n)
            seen.add(n)

    # Build the name clause
    name_clause = " OR ".join(
        f'"{n}"[Title/Abstract]' for n in names[:6]  # cap to keep query compact
    )

    # 2) Keywords from the question (stop-word filtered)
    words = re.findall(r"[A-Za-z0-9+\-/]+", user_q)

    stop = {
        "what", "known", "about", "this", "compound", "can", "could", "would",
        "there", "any", "info", "information", "study", "studies",
        "is", "are", "was", "were", "has", "have", "had", "been",
        "do", "does", "did", "of", "for", "in", "on", "with", "to",
        "or", "and", "the", "a", "an", "it", "its"
    }

    keywords = [
        w for w in words
        if len(w) >= 3 and w.lower() not in stop
    ]

    if not keywords:
        keywords = ["pharmacology"]

    kw_clause = " OR ".join(
        f'"{w}"[Title/Abstract]' for w in dict.fromkeys(keywords)
    )

    # 3) Join clauses
    clauses = []
    if name_clause:
        clauses.append(name_clause)
    clauses.append(kw_clause)

    query = " AND ".join(f"({c})" for c in clauses)
    return query

def format_articles_for_prompt(arts: list, abstracts: dict, limit=5, abs_chars=600):
    lines = []
    for a in arts[:limit]:
        pmid = a["pmid"]
        line = f"- {a['title']} ({a['journal']}, {a['year']}); PMID:{pmid}"
        if a.get("doi"):
            line += f"; DOI:{a['doi']}"
        abs_txt = abstracts.get(pmid, "")
        if abs_txt:
            line += f"\n  Abstract: {abs_txt[:abs_chars]}"
        lines.append(line)
    return "\n".join(lines) if lines else "No directly relevant PubMed items found."

# =========================
# Heuristic classification
# =========================
def classify_druglikeness(desc: dict):
    """
    Assign a heuristic physicochemical profile category using
    Lipinski-style descriptor limits, Veber-style TPSA and
    rotatable-bond criteria, and ESOL-predicted aqueous solubility.

    Categories:
      A. Favorable physicochemical profile
      B. Intermediate physicochemical profile
      C. Physicochemical limitations
      D. Out of scope due to missing or non-numeric required data

    QED is calculated and reported separately and does not determine
    the category.

    These categories describe computational physicochemical profiles
    and should not be interpreted as experimental measurements of
    absorption, permeability, oral bioavailability, efficacy, or safety.
    """
    needed = ["Molecular Weight", "LogP", "HBA", "HBD", "TPSA", "Rotatable Bonds", "LogS (ESOL)"]
    if not all(k in desc and isinstance(desc[k], (int, float)) for k in needed if k != "LogS (ESOL)"):
        return "D. Out of scope"
    logS_val = desc.get("LogS (ESOL)")
    if isinstance(logS_val, str):
        return "D. Out of scope"

    # Lipinski rules
    violations = 0
    violations += 1 if desc["Molecular Weight"] > 500 else 0
    violations += 1 if desc["LogP"] > 5 else 0
    violations += 1 if desc["HBA"] > 10 else 0
    violations += 1 if desc["HBD"] > 5 else 0

    tpsa = desc["TPSA"]
    rb = desc["Rotatable Bonds"]

        # --- Base classification ---
    if violations == 0 and tpsa <= 140 and rb <= 10 and logS_val > -4.5:
        bucket = "A. Favorable physicochemical profile"
    elif violations <= 1 and tpsa <= 160 and rb <= 12 and (-6 < logS_val <= -4.5 or violations == 1):
        bucket = "B. Intermediate physicochemical profile"
    elif violations >= 2 or tpsa > 180 or logS_val <= -6:
        bucket = "C. Physicochemical limitations"
    else:
        bucket = "B. Intermediate physicochemical profile"

    return bucket

def gpt_pk_interpretation(row: dict) -> str:
    """
    Generate a scientifically conservative interpretation of the
    physicochemical profile calculated/retrieved by SPARK.

    The interpretation describes descriptor-based trends only and
    does not treat Lipinski, Veber, QED, TPSA, or ESOL predictions
    as experimental measurements of permeability, absorption, or
    oral bioavailability.
    """
    # Deterministic Lipinski/Veber threshold checks.
    # Python performs the numerical comparisons so GPT only interprets them.
    mw = row.get("Molecular Weight")
    logp = row.get("LogP")
    hba = row.get("HBA")
    hbd = row.get("HBD")
    tpsa = row.get("TPSA")
    rb = row.get("Rotatable Bonds")

    def threshold_status(value, limit):
        if not isinstance(value, (int, float)):
            return "Not available"
        return "Pass" if value <= limit else "Fail"

    threshold_checks = {
        "MW <= 500": threshold_status(mw, 500),
        "LogP <= 5": threshold_status(logp, 5),
        "HBA <= 10": threshold_status(hba, 10),
        "HBD <= 5": threshold_status(hbd, 5),
        "TPSA <= 140": threshold_status(tpsa, 140),
        "Rotatable Bonds <= 10": threshold_status(rb, 10),
    }

    threshold_text = "\n".join(
        f"{rule}: {status}"
        for rule, status in threshold_checks.items()
    )

    prompt = f"""
Compound: {row.get('Display Name') or row.get('Compound')}
SMILES: {row.get('Input SMILES') or row.get('SMILES')}

Molecular Weight: {row.get('Molecular Weight')}
LogP: {row.get('LogP')}
TPSA: {row.get('TPSA')}
HBA: {row.get('HBA')}
HBD: {row.get('HBD')}
Rotatable Bonds: {row.get('Rotatable Bonds')}
Solubility (LogS, ESOL): {row.get('LogS (ESOL)')}
Solubility Category: {classify_solubility(row.get('LogS (ESOL)'))}
QED: {row.get('QED')}
pKa: {row.get('pKa')}
pKa: {row.get('pKa Source')}
SPARK Category: {row.get('Category')}

Deterministic Lipinski/Veber threshold checks:
{threshold_text}

Interpret this compound's physicochemical profile using ONLY the
values and deterministic threshold checks provided above.

Important scientific rules:

1. Lipinski and Veber criteria are descriptor-based guidelines.
   They do NOT experimentally demonstrate oral bioavailability,
   intestinal absorption, membrane permeability, efficacy, or safety.

2. TPSA may be discussed as a molecular polarity descriptor.
   Do NOT state that TPSA proves good or poor permeability.

3. ESOL LogS is a computational estimate of aqueous solubility.
   Clearly identify it as predicted solubility, not an experimental
   solubility measurement.

4. LogP describes calculated/reported lipophilicity. Do not claim
   that a particular LogP value proves oral absorption or
   bioavailability.

5. QED is a drug-likeness metric. It is not a probability that the
   compound will become a drug and does not establish bioavailability. Do not describe QED using probability or likelihood language.

6. Use the supplied Rotatable Bonds value exactly. Do not state that
   rotatable-bond information is unavailable when a value is provided.

7. Do not invent missing data. If a value is None, Not found,
   unavailable, or otherwise missing, explicitly state that it is
   unavailable and do not infer it from the molecular structure.

8. Do not describe the compound as an "optimal oral candidate,"
   "ideal oral drug," "promising oral candidate," or claim that it
   has "good bioavailability" based only on these descriptors.

9. Use the supplied deterministic Lipinski/Veber threshold checks
   exactly as provided. Do not independently recalculate, reverse,
   reinterpret, or contradict the Pass/Fail results. Describe these
   checks as consistency with descriptor-based guidelines rather
   than proof of drug performance.

10. Distinguish predicted/calculated properties from experimentally
    measured pharmacokinetic behavior.

11. Use the supplied Solubility Category exactly when describing
    qualitative solubility. Do not independently assign qualitative
    solubility terms from the ESOL value.

12. When comparing a numeric descriptor with a threshold, explicitly
    verify the numerical relationship before describing it as above
    or below the threshold. For example, 63.6 is below 140.

13. Do not state or imply that LogP or TPSA demonstrates, predicts,
    improves, or is favorable for membrane permeability. Discuss
    LogP as lipophilicity and TPSA as molecular polarity only.

14. If pKa Source is "GPT estimate", explicitly identify the pKa as
    an AI-generated estimate and do not use it to infer ionization,
    absorption, distribution, or other pharmacokinetic behavior.
Provide four short sections:

1. Physicochemical Profile
Summarize MW, LogP, TPSA, HBA, HBD, rotatable bonds, QED, and ESOL
LogS without overstating their biological implications.

2. Lipinski and Veber Assessment
State which supplied descriptor thresholds are satisfied or exceeded.
Use the actual values provided.

3. Solubility and Lipophilicity Considerations
Discuss the ESOL prediction and LogP cautiously. Identify ESOL as a
prediction.

4. Limitations
Briefly explain what cannot be concluded from these descriptors alone,
including actual permeability, absorption, oral bioavailability,
metabolism, efficacy, and safety.
"""

    return chat_with_gpt(
        prompt,
        system=(
            "You are an expert in medicinal chemistry and ADME. "
            "Interpret computational physicochemical descriptors "
            "conservatively. Distinguish descriptor-based guidelines "
            "and predictions from experimentally measured "
            "pharmacokinetic properties. Never invent missing data."
        )
    )

def gpt_predict_pka_concise(compound_or_smiles: str):
    prompt = f"""Return a single-line predicted pKa (or "none expected") for:
{compound_or_smiles}

Rules:
- If no ionizable groups expected near physiological range, answer exactly: none expected
- Otherwise give a single numeric value (one number), e.g., 10.3
- No prose, no units, just the value or "none expected"
"""
    ans = chat_with_gpt(prompt)

    # sanitize
    return ans.splitlines()[0].strip()

def answer_with_literature(
    matched_row: dict,
    user_q: str,
    mindate="2015",
    retmax=12,
    max_items=5
) -> tuple[str, str]:
    """
    Returns (answer_text, citations_block).

    Answer is primarily grounded in the fetched articles, but may also use
    well-established background knowledge, clearly separated from evidence.
    """

    query = build_generic_literature_query(
        matched_row,
        user_q
    )

    print(f"\nPubMed query:\n{query}\n")

    pmids = pubmed_esearch(
        query,
        retmax=retmax,
        sort="bestmatch",
        mindate=mindate
    )

    summaries = pubmed_fetch_summaries(pmids)

    abstracts = pubmed_fetch_abstracts(
        pmids[:max_items]
    )

    articles_block = format_articles_for_prompt(
        summaries,
        abstracts,
        limit=max_items
    )

    # Compact compound context
    keys = [
        "Compound",
        "SMILES",
        "Molecular Formula",
        "Molecular Weight",
        "LogP",
        "LogS (ESOL)",
        "TPSA",
        "pKa",
        "QED",
        "Category"
    ]

    ctx = "\n".join(
        f"{k}: {matched_row.get(k, 'Not found')}"
        for k in keys
    )

    prompt = f"""You are a medicinal chemistry assistant.

You have:
1) Compound data.
2) A small set of PubMed articles (titles/abstracts).
3) Your general background knowledge in pharmacology and medicinal chemistry.

Your task:
- FIRST, extract evidence-based findings that are directly supported by
  the provided PubMed articles and clearly relate to the user's question.

- THEN, add a short section with general background / interpretation
  where you may use well-established knowledge about similar compounds,
  targets, or mechanisms.

- ALWAYS distinguish clearly between:
  - "Evidence-based (from the articles)"
  - "General background / plausible interpretation"

Compound data:
{ctx}

User question:
{user_q}

Articles:
{articles_block}

Instructions:
- Do not claim that an article supports something unless that information
  appears in the supplied article information or abstract.
- Do not invent experimental results, mechanisms, targets, or citations.
- Clearly identify uncertainty.
- Keep evidence from PubMed separate from general interpretation.
- When discussing computational SPARK descriptors, do not treat them as
  experimentally measured pharmacokinetic properties.
"""

    answer = chat_with_gpt(
        prompt,
        system=(
            "You are a medicinal chemistry research assistant. "
            "Distinguish evidence from interpretation and never invent "
            "literature findings or citations."
        )
    )

    citations = []

    for article in summaries[:max_items]:

        citation = (
            f"{article.get('title', 'Untitled')} "
            f"({article.get('journal', '')}, "
            f"{article.get('year', '')}) "
            f"PMID: {article.get('pmid', '')}"
        )

        if article.get("doi"):
            citation += f"; DOI: {article['doi']}"

        citations.append(citation)

    citations_block = (
        "\n".join(citations)
        if citations
        else "No directly relevant PubMed citations found."
    )

    return answer, citations_block

# =========================
# Core processing
# =========================
def process_compound(entry: str) -> dict:
    result = {
        "Compound": entry,
        "Input SMILES": entry if looks_like_smiles(entry) else "Not applicable",
        "Structure Valid": "Not checked",
        "PubChem Match Status": "Not searched",
        "PubChem SMILES": "Not found",
        "Display Name": "Not identified",
        "Molecular Formula": "Not found",
        "Molecular Weight": "Not found",
        "Exact Mass": "Not found",
        "SMILES": "Not found",
        "LogP": "Not found",
        "TPSA": "Not found",
        "HBA": "Not found",
        "HBD": "Not found",
        "Rotatable Bonds": "Not found",
        "LogS (ESOL)": "Not found",
        "pKa": "Not found",
        "pKa Source": "Not available",
        "Melting Point": "Not found",
        "Boiling Point": "Not found",
    }
    names = {}            # will hold PubChem names/synonyms if we get a CID
    chembl_fields = {}    # ensure defined before any checks

    is_smiles_input = looks_like_smiles(entry)

    if is_smiles_input:

        prepared_smiles = prepare_smiles_for_lookup(entry)

        if prepared_smiles["valid"]:
            result["Structure Valid"] = "Yes"
        else:
            result["Structure Valid"] = "No"

    else:
        result["Structure Valid"] = "Not applicable"

    cid, pubchem_status = get_pubchem_cid(
        entry,
        is_smiles_input
    )

    result["PubChem Match Status"] = pubchem_status

    # ---------------------------------------------------------
    # Retrieve core properties directly from PubChem
    # ---------------------------------------------------------

    pubchem_smiles = None
    pubchem_props = {}

    if cid:

        # Store CID so we can trace every result back to PubChem
        result["PubChem CID"] = cid

        pubchem_props = get_properties_by_cid(cid)

        if pubchem_props:

            # PubChem is the primary source for these properties.
            # This also gives us the PubChem SMILES needed by RDKit.
            pubchem_smiles = pubchem_props.get("SMILES")
            if pubchem_smiles:
                result["PubChem SMILES"] = pubchem_smiles
            # Merge PubChem properties into the result
            result.update(pubchem_props)

        else:
            print(
                f"⚠️ PubChem CID {cid} was found for {entry}, "
                f"but no core property table was returned."
            )

    else:
        print(
            f"⚠️ No PubChem CID found for: {entry}"
        )

        # ---------------------------------------------------------
    # Select the structure analyzed by RDKit
    # ---------------------------------------------------------
    #
    # If SPARK receives a SMILES as input, that supplied
    # structure remains the primary structure for calculation.
    #
    # PubChem SMILES is retained separately as an external
    # database reference and does not replace the user's
    # supplied molecular structure.
    # ---------------------------------------------------------

    if is_smiles_input:
        smiles_to_use = entry
    else:
        smiles_to_use = pubchem_smiles

        # ---------------------------------------------------------
    # Calculate RDKit descriptors from the PubChem structure
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # PubChem remains the primary source for properties that
    # PubChem reports.
    #
    # RDKit is used for:
    #   1. QED
    #   2. ESOL LogS
    #   3. independent validation values
    #
    # RDKit must NOT silently overwrite PubChem values.
    # ---------------------------------------------------------

    if smiles_to_use:

        rd = calculate_rdkit_descriptors_full(smiles_to_use)

        if rd:

            # Properties that SPARK calculates because they are
            # not being taken directly from PubChem here
            result["QED"] = rd.get("QED")
            result["LogS (ESOL)"] = rd.get("LogS (ESOL)")

            # --------------------------------------------------
            # Keep RDKit values separately for validation
            # --------------------------------------------------

            result["RDKit Molecular Formula"] = rd.get("Molecular Formula")
            result["RDKit Molecular Weight"] = rd.get("Molecular Weight")
            result["RDKit Exact Mass"] = rd.get("Exact Mass")
            result["RDKit LogP"] = rd.get("LogP")
            result["RDKit TPSA"] = rd.get("TPSA")
            result["RDKit HBA"] = rd.get("HBA")
            result["RDKit HBD"] = rd.get("HBD")
            result["RDKit Rotatable Bonds"] = rd.get("Rotatable Bonds")

            # --------------------------------------------------
            # Fallbacks
            # --------------------------------------------------
            # These are only used if PubChem genuinely did not
            # provide the corresponding value.
            # --------------------------------------------------

            fallback_fields = [
                "Molecular Formula",
                "Molecular Weight",
                "Exact Mass",
                "LogP",
                "TPSA",
                "HBA",
                "HBD",
                "Rotatable Bonds",
            ]

            for field in fallback_fields:

                if result.get(field) in (
                    None,
                    "",
                    "Not found",
                    "Invalid SMILES"
                ):

                    if rd.get(field) is not None:
                        result[field] = rd[field]

    # ---------------------------------------------------------
    # Pull detailed PubChem properties using ONE PUG View request
    # ---------------------------------------------------------

    if cid:

        detailed = get_detailed_properties(cid)

        melting_point = detailed.get("Melting Point")

        if melting_point not in (
            None,
            "",
            "Not found"
        ):
            result["Melting Point"] = melting_point

        boiling_point = detailed.get("Boiling Point")

        if boiling_point not in (
            None,
            "",
            "Not found"
        ):
            result["Boiling Point"] = boiling_point

        pka_text = detailed.get("pKa")

        if pka_text not in (
            None,
            "",
            "Not found"
        ):
            result["pKa"] = pka_text
            result["pka Source"] = "PubChem"

        # Capture PubChem names/synonyms for literature queries
        names = get_pubchem_names_and_synonyms(
            cid,
            core_properties=pubchem_props
        )

        if names.get("title"):
            result["PubChem Title"] = names["title"]
            result["Display Name"] = names["title"]

        if names.get("iupac"):
            result["IUPAC Name"] = names["iupac"]

        if names.get("inchikey"):
            result["InChIKey"] = names["inchikey"]

        if names.get("synonyms"):
            result["Synonyms (PubChem)"] = "; ".join(
                names["synonyms"][:10]
            )

    # Enrich with ChEMBL

    if names.get("title"):
        chembl_query = names["title"]

    elif not is_smiles_input:
        chembl_query = entry

    else:
        chembl_query = smiles_to_use

    chembl_rec = chembl_search_molecule(chembl_query)
    chembl_fields = chembl_extract_props(chembl_rec)

    # ---------------------------------------------------------
    # Verify ChEMBL identity using structural InChIKey
    # ---------------------------------------------------------
    #
    # For SMILES input, the InChIKey generated directly from
    # the supplied structure is used as the primary reference.
    #
    # If the input was a name, PubChem's InChIKey is used.
    # ---------------------------------------------------------

    input_inchikey = None

    if is_smiles_input:

        prepared_identity = prepare_smiles_for_lookup(entry)

        if prepared_identity["valid"]:
            input_inchikey = prepared_identity.get("inchikey")

    pubchem_key = result.get("InChIKey")
    chembl_key = chembl_fields.get("ChEMBL InChIKey")

    # Supplied structure takes priority for SMILES input.
    reference_key = input_inchikey or pubchem_key

    if reference_key and chembl_key:

        if reference_key.upper() != chembl_key.upper():

            print(
                f"\n⚠️ ChEMBL structure mismatch for {entry}"
                f"\n   Reference InChIKey: {reference_key}"
                f"\n   ChEMBL InChIKey:    {chembl_key}"
                f"\n   ChEMBL result rejected."
            )

            chembl_fields = {}
            chembl_rec = {}

        else:

            print(
                f"✅ ChEMBL structure verified by InChIKey: "
                f"{reference_key}"
            )

    elif chembl_fields and not chembl_key:

        print(
            f"⚠️ ChEMBL record has no InChIKey for {entry}. "
            f"ChEMBL result rejected because structural "
            f"identity could not be verified."
        )

        chembl_fields = {}
        chembl_rec = {}

    elif chembl_fields and not reference_key:

        print(
            f"⚠️ No structural InChIKey available to verify "
            f"ChEMBL result for {entry}. "
            f"ChEMBL result rejected."
        )

        chembl_fields = {}
        chembl_rec = {}

    if chembl_fields:
    

        # Merge the ChEMBL fields into the result (namespaced)
        result.update(chembl_fields)
        if (
            result.get("Display Name") == "Not identified"
            and chembl_fields.get("ChEMBL Pref Name")
        ):
            result["Display Name"] = chembl_fields["ChEMBL Pref Name"]

        # Optional: bioactivity summary
        chembl_id = chembl_fields.get("ChEMBL ID")
        if chembl_id:
            result.update(chembl_bioactivity_summary(chembl_id))

       
    # pKa fallback if still not found (concise)
    if result["pKa"] == "Not found":
        if result.get("Structure Valid") == "No":
            result["pKa"] = "Not available"
            result["pKa Source"] = "Invalid structure"
        else:
            result["pKa"] = gpt_predict_pka_concise(
            result.get("SMILES") if result.get("SMILES") != "Not found" else entry
            )
            result["pKa Source"] = "GPT estimate"

    # Convert numeric strings -> floats where possible
    for fld in ["Molecular Weight", "Exact Mass", "LogP", "TPSA", "HBA", "HBD", "Rotatable Bonds", "LogS (ESOL)"]:
        v = result.get(fld)
        if isinstance(v, str) and v not in ("Not found", "Invalid SMILES"):
            try:
                result[fld] = float(v)
            except Exception:
                pass

    result["Category"] = classify_druglikeness(result)
    result["GPT Interpretation"] = gpt_pk_interpretation(result)
    return result

# =========================
def load_compounds_interactive() -> list[str]:
    """
    Ask the user to enter compounds manually or via a spreadsheet.
    Supports: .xlsx/.xls (Excel) and .csv.
    Returns a list of strings (names or SMILES), deduped in original order.
    """
    def dedupe_preserve_order(items):
        seen = set()
        out = []
        for it in items:
            if it not in seen:
                out.append(it)
                seen.add(it)
        return out

    choice = input("Do you want to enter compounds manually or via file? (manual/excel/csv): ").strip().lower()

    if choice in ("excel", "csv"):
        file_path = input("Enter the path to the file: ").strip().strip('"')
        p = pathlib.Path(file_path)

        if not p.exists():
            print(f"❌ File not found: {p}")
            return []

        # Ask column name
        column_name = input("Enter the column name that contains the compound names/SMILES: ").strip()

        try:
            if p.suffix.lower() in (".xlsx", ".xls"):
                # optional sheet name or index
                sheet = input("Enter sheet name (or press Enter for first sheet): ").strip()
                sheet_arg = 0 if sheet == "" else sheet
                df = pd.read_excel(p, sheet_name=sheet_arg)
            elif p.suffix.lower() == ".csv":
                # Optional delimiter prompt (defaults to comma)
                delim = input("CSV delimiter? (press Enter for ','): ").strip()
                df = pd.read_csv(p, sep=delim or ",")
            else:
                print("❌ Unsupported file type. Use .xlsx, .xls, or .csv")
                return []

            if column_name not in df.columns:
                print(f"❌ Column '{column_name}' not found. Available columns:\n  - " + "\n  - ".join(map(str, df.columns)))
                return []

            compounds = df[column_name].dropna().astype(str).str.strip()
            compounds = [c for c in compounds if c]  # remove blanks
            compounds = dedupe_preserve_order(compounds)
            print(f"✅ Loaded {len(compounds)} compounds from {p.name}.")
            return compounds

        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []

    # Manual input fallback
    user = input("Enter SMILES of compounds (comma-separated): ").strip()
    compounds = [x.strip() for x in user.split(",") if x.strip()]
    compounds = dedupe_preserve_order(compounds)
    print(f"✅ Loaded {len(compounds)} compounds from manual input.")
    return compounds

# CLI
# =========================
if __name__ == "__main__":
    compounds = load_compounds_interactive()
    results = []

    # --- Core processing: PubChem + RDKit + ChEMBL ---
    for c in compounds:
        print(f"\nProcessing: {c}")
        row = process_compound(c)
        for k, v in row.items():
            print(f"{k}: {v}")
        results.append(row)

    # === Rule-based physicochemical grouping ===
    for compound in results:
        # Basic fields
        mw = compound.get("Molecular Weight")
        logp = compound.get("LogP")
        hbd = compound.get("HBD")
        hba = compound.get("HBA")
        tpsa = compound.get("TPSA")
        rb = compound.get("Rotatable Bonds")
        logS = compound.get("LogS (ESOL)")
        smiles = compound.get("SMILES")

        # Lipinski violations
        violations = 0
        if isinstance(mw, (int, float)) and mw > 500:
            violations += 1
        if isinstance(logp, (int, float)) and logp > 5:
            violations += 1
        if isinstance(hbd, (int, float)) and hbd > 5:
            violations += 1
        if isinstance(hba, (int, float)) and hba > 10:
            violations += 1

        compound["Lipinski Violations"] = violations

        # Veber criteria
        passes_veber = (
            isinstance(tpsa, (int, float)) and tpsa <= 140
            and isinstance(rb, (int, float)) and rb <= 10
        )
        compound["Passes Veber"] = passes_veber

        # Heteroatoms
        if isinstance(smiles, str) and smiles not in ("Not found", "Invalid SMILES", ""):
            compound["Heteroatoms"] = count_heteroatoms(smiles)
        else:
            compound["Heteroatoms"] = 0

        # Predicted solubility category
        compound["Solubility Category"] = classify_solubility(logS)

        heteroatoms = compound["Heteroatoms"]
        solubility = compound["Solubility Category"]

        if heteroatoms >= 6 and solubility == "Moderately soluble":
            adjusted_solubility = "Highly soluble"
        elif heteroatoms <= 2 and solubility == "Moderately soluble":
            adjusted_solubility = "Poorly soluble"
        else:
            adjusted_solubility = solubility

        # Rule-based physicochemical group assignment
        required_values = [mw, logp, hbd, hba, tpsa, rb, logS]
        if (
            compound.get("Structure Valid") == "No"
            or not all(isinstance(v, (int, float)) for v in required_values)
        ):
            pk_group = "Out of scope: insufficient valid descriptor data"
        elif violations == 0 and passes_veber:
            pk_group = "Group 1: Favorable descriptor profile"
        elif violations <= 1 and adjusted_solubility != "Poorly soluble":
            pk_group = "Group 2: Intermediate descriptor profile"
        elif violations <= 2 and adjusted_solubility == "Poorly soluble":
            pk_group = "Group 3: Predicted solubility limitation"
        else:
            pk_group = "Group 4: Multiple descriptor limitations"

        compound["PK Group"] = pk_group

        # === QED-based descriptor assessment (non-gating) ===
        qed = compound.get("QED")

        if isinstance(qed, (int, float)):
            if (
                pk_group == "Group 1: Favorable descriptor profile"
                and qed >= 0.7
            ):
                assessment = "Higher QED descriptor profile"

            elif (
                pk_group in (
                    "Group 1: Favorable descriptor profile",
                    "Group 2: Intermediate descriptor profile"
                )
                and qed >= 0.5
            ):
                assessment = "Moderate QED descriptor profile"

            elif pk_group == "Group 3: Predicted solubility limitation":
                assessment = "Predicted solubility limitation"

            else:
                assessment = "Lower QED descriptor profile"

        else:
            assessment = "QED assessment unavailable"

        compound["Descriptor Assessment"] = assessment
        # Ensure keys exist for CSV and downstream logic
        compound.setdefault("Group Summary", "")
        compound.setdefault("Follow-up Responses", "")

    # === 🧪 GPT-Based Compound Grouping (global summary) ===
    compound_summaries = ""
    for comp in results:
        compound_summaries += f"""
Display Name: {comp.get('Display Name')}
Input SMILES: {comp.get('Input SMILES')}
PubChem Match Status: {comp.get('PubChem Match Status')}
Molecular Weight: {comp.get('Molecular Weight')}
LogP: {comp.get('LogP')}
LogS (ESOL): {comp.get('LogS (ESOL)')}
Solubility Category: {comp.get('Solubility Category')}
TPSA: {comp.get('TPSA')}
pKa: {comp.get('pKa')}
pKa Source: {comp.get('pKa Source')}
HBD: {comp.get('HBD')}
HBA: {comp.get('HBA')}
PK Group: {comp.get('PK Group')}
"""

    group_prompt = f"""
You are a medicinal chemistry research assistant.

The following compounds have been evaluated using physicochemical
descriptors and rule-based grouping in SPARK.

{compound_summaries}

Interpret the supplied results conservatively.

Important scientific rules:

1. The SPARK groups describe physicochemical descriptor profiles.
   They are not rankings of clinical potential, efficacy, safety,
   or probability of becoming a drug.

2. Lipinski and Veber criteria are descriptor-based guidelines.
   Passing these criteria does not experimentally establish oral
   bioavailability, absorption, permeability, efficacy, or safety.

3. ESOL LogS is a computational prediction of aqueous solubility.
   Clearly describe it as predicted solubility rather than an
   experimentally measured value.

4. LogP describes lipophilicity and should not be interpreted by
   itself as proof of absorption or bioavailability.

5. TPSA is a molecular polarity descriptor. Do not state that TPSA
   alone proves good or poor membrane permeability.

6. Do not describe compounds as:
   - optimal drugs
   - ideal oral candidates
   - promising drug candidates
   - high-confidence leads
   - clinically promising
   - having good bioavailability

   unless independent experimental evidence supporting such a
   statement has explicitly been supplied.

7. Do not invent missing experimental data.

8. Use the supplied PK Group exactly as assigned by SPARK. Do not
   independently move compounds between groups.

9. Compound identity must come only from the supplied Display Name.
   If Display Name is "Not identified", refer to the compound as
   "the supplied structure" or "the supplied SMILES". Do NOT infer,
   guess, or generate a chemical name from the SMILES.

10. Use the supplied Solubility Category when describing qualitative
    solubility. Do not independently assign terms such as highly
    soluble, moderately soluble, poorly soluble, or very poorly soluble
    from the ESOL value.

11. ESOL LogS is a computational prediction. Any supplied Solubility
    Category is therefore also a predicted classification rather than
    an experimentally measured solubility classification.

12. Do not infer membrane permeability from LogP, TPSA, molecular
    weight, hydrogen-bond counts, or the SPARK group.

13. If pKa Source is "GPT estimate", explicitly describe the pKa as an
    AI-generated estimate rather than an experimental or database value.

14. Do not infer ionization, absorption, distribution, permeability,
    bioavailability, efficacy, or safety from an AI-estimated pKa alone.

For each compound, briefly explain:
- its assigned SPARK group,
- which supplied physicochemical descriptors contribute to that group,
- predicted solubility considerations where relevant,
- and important limitations of interpreting these descriptors.

Finish with a short overall summary describing differences among the
descriptor profiles without ranking compounds by therapeutic or
clinical potential.
"""

    grouping_analysis = chat_with_gpt(group_prompt, model="gpt-4o")
    print("\n🧠 GPT Grouping & Explanation:\n")
    print(grouping_analysis)

    # Save grouping summary to a text file
    grouping_filename = f"grouping_summary_{datetime.now().strftime('%Y-%m-%d_%H%M')}.txt"
    with open(grouping_filename, "w", encoding="utf-8") as f:
        f.write(grouping_analysis)
    print(f"✅ Grouping summary saved to: {grouping_filename}")

    # Attach grouping analysis as a special row for CSV (optional, like before)
    results.append({
        "Compound": "GROUPING ANALYSIS",
        "Group Summary": grouping_analysis
    })

    # === Write single CSV with all fields ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    csv_file = f"steroid_properties_{timestamp}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [

    # --- SPARK input / structural identity ---
    "Compound",
    "Input SMILES",
    "Structure Valid",
    "Display Name",

    # --- PubChem identity ---
    "PubChem Match Status",
    "PubChem CID",
    "PubChem Title",
    "IUPAC Name",
    "InChIKey",
    "PubChem SMILES",
    "PubChem Connectivity SMILES",
    "Synonyms (PubChem)",

    # --- Main physicochemical properties ---
    "Molecular Formula",
    "Molecular Weight",
    "Exact Mass",
    "SMILES",
    "LogP",
    "LogS (ESOL)",
    "TPSA",
    "HBA",
    "HBD",
    "Rotatable Bonds",
    "pKa",
    "Melting Point",
    "Boiling Point",
    "QED",

    # --- RDKit validation properties ---
    "RDKit Molecular Formula",
    "RDKit Molecular Weight",
    "RDKit Exact Mass",
    "RDKit LogP",
    "RDKit TPSA",
    "RDKit HBA",
    "RDKit HBD",
    "RDKit Rotatable Bonds",

    # --- Rule-based PK fields ---
    "Lipinski Violations",
    "Passes Veber",
    "Heteroatoms",
    "Solubility Category",
    "PK Group",
    "Descriptor Assessment",
    "Group Summary",
    "Follow-up Responses",

    # --- ChEMBL extras ---
    "ChEMBL ID",
    "ChEMBL Pref Name",
    "ChEMBL SMILES",
    "ChEMBL InChIKey",
    "ChEMBL QED",
    "ChEMBL ALogP",
    "ChEMBL MW Freebase",
    "ChEMBL PSA",
    "ChEMBL HBA",
    "ChEMBL HBD",
    "ChEMBL RO5 Violations",
    "ChEMBL ATC Codes",
    "ChEMBL Bioactivity Count",
    "ChEMBL Max pChEMBL",

    # --- Final analysis ---
    "Category",
    "GPT Interpretation",
]

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            if 'Exact Mass"' in row:
                row['Exact Mass'] = row.pop('Exact Mass"')
            writer.writerow(row)

    print(f"\n✅ All data saved to: {csv_file}")

    # === Interactive GPT Q&A (literature-grounded) ===
    def pick_compound(results, target):
        for r in results:
            if r.get("Compound") == target or r.get("SMILES") == target:
                return r
        matches = [r for r in results if target.lower() in str(r.get("Compound","")).lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print("\nMultiple matches:")
            for i, r in enumerate(matches, 1):
                print(f"  {i}) {r.get('Compound')}  [{r.get('SMILES')}]")
            try:
                idx = int(input("Pick a number: ").strip())
                if 1 <= idx <= len(matches):
                    return matches[idx-1]
            except Exception:
                pass
        return None

    while True:
        follow_up = input("\n💬 Ask a question about any compound (or type 'exit'): ").strip()
        if follow_up.lower() == "exit":
            break

        target = input("Which compound? (Name or SMILES): ").strip()
        matched = pick_compound(results, target)

        if not matched:
            print("❌ Compound not found. Please try again.")
            continue

        # Try literature-grounded answer
        try:
            ans, cites = answer_with_literature(matched, follow_up, mindate="2010", retmax=20, max_items=6)
            print("\n🧠 Literature-grounded answer:\n", ans)
            print("\n📚", cites)
        except Exception as e:
            # Graceful fallback: property-based answer only
            print(f"\n⚠️ PubMed retrieval failed ({e}). Falling back to property-based answer.")
            keys = ["Compound","SMILES","Molecular Formula","Molecular Weight","LogP","LogS (ESOL)","TPSA","pKa","QED","Category"]
            ctx = "\n".join(f"{k}: {matched.get(k,'Not found')}" for k in keys)
            prompt = f"""You are a medicinal chemistry assistant.

Compound data:
{ctx}

User question:
{follow_up}

Answer concisely and reference the numbers from the compound data."""
            ans = chat_with_gpt(prompt)
            cites = "Citations: none (literature fetch failed)."

        matched.setdefault("Follow-up Responses", "")
        matched["Follow-up Responses"] += f"\nQ: {follow_up}\nA: {ans}\n{cites}\n"

    # === (Optional) ALL-compounds question, literature-aware ===
    user_question = input("\n🌐 Ask a question about ALL compounds (or press Enter to skip): ").strip()
    if user_question:
        # Use the first 1-2 names to bias PubMed, but still answer generally
        seed = next((r for r in results if r.get("Compound") and r.get("Compound") != "GROUPING ANALYSIS"), None)
        try:
            ans, cites = answer_with_literature(seed or {}, user_question, mindate="2015", retmax=15, max_items=6)
            print("\n🧠 Literature-grounded answer (ALL):\n", ans)
            print("\n📚", cites)
        except Exception as e:
            bundle = [{k: c.get(k) for k in [
                "Compound","SMILES","Molecular Formula","Molecular Weight","Exact Mass",
                "LogP","LogS (ESOL)","TPSA","HBA","HBD","Rotatable Bonds","pKa","QED","Category",
                "PK Group"
            ]} for c in results if c.get("Compound") != "GROUPING ANALYSIS"]
            prompt = f"""You are a medicinal chemistry assistant.

Here are compounds and their properties (JSON-like list):
{bundle}

Question: {user_question}

In your answer, compare compounds and reference specific numbers where helpful."""
            ans = chat_with_gpt(prompt)
            print("\n🧠 GPT Answer (fallback):\n", ans)
