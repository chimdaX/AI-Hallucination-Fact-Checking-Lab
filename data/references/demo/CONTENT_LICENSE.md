# Content licence and attribution: `demo` reference collection

The **code** in this repository is MIT-licensed. The **documents in this folder are not**.
Each one keeps its original licence, listed below.

## Wikipedia and Wikiquote texts: CC BY-SA 4.0

Files named `wikipedia_*.txt` and `wikiquote_*.txt` are plain-text extracts of the
listed page revisions. Their authors are the Wikipedia/Wikiquote contributors (see each
page's history). They are licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-sa/4.0/).

**Changes made:** the text was produced by the MediaWiki TextExtracts API (`explaintext`),
which leaves out infoboxes, tables, images, citations, and some formatting. The extract
was stored without further edits. If you redistribute these files or adaptations of them,
you must keep this attribution and licence (share-alike).

## NASA Mars Fact Sheet

`nasa_mars_fact_sheet.txt` was converted from the NSSDCA web page by
`scripts/build_demo_collection.py`: table cells are joined with ` | `, superscripts are
written as `^`, and the page-footer contact block is omitted. NASA content is generally
not subject to copyright in the United States
([NASA media usage guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/)).
**This status was inferred and should be confirmed manually** before any redistribution
beyond this project.

## Documents

| File | Title | Author / organization | Version | Source | Licence |
|---|---|---|---|---|---|
| `wikipedia_eiffel_tower.txt` | Eiffel Tower (Wikipedia) | Wikipedia contributors | revision 1376706466 (2026-09-25T19:49:48Z) | [link](https://en.wikipedia.org/w/index.php?title=Eiffel_Tower&oldid=1376706466) | CC BY-SA 4.0 |
| `wikipedia_mount_everest.txt` | Mount Everest (Wikipedia) | Wikipedia contributors | revision 1376942985 (2026-09-27T02:29:28Z) | [link](https://en.wikipedia.org/w/index.php?title=Mount_Everest&oldid=1376942985) | CC BY-SA 4.0 |
| `wikipedia_properties_of_water.txt` | Properties of water (Wikipedia) | Wikipedia contributors | revision 1377952885 (2026-10-02T02:56:56Z) | [link](https://en.wikipedia.org/w/index.php?title=Properties_of_water&oldid=1377952885) | CC BY-SA 4.0 |
| `wikipedia_sunlight.txt` | Sunlight (Wikipedia) | Wikipedia contributors | revision 1378209366 (2026-10-03T14:04:31Z) | [link](https://en.wikipedia.org/w/index.php?title=Sunlight&oldid=1378209366) | CC BY-SA 4.0 |
| `wikipedia_leap_year.txt` | Leap year (Wikipedia) | Wikipedia contributors | revision 1378838087 (2026-10-06T11:54:42Z) | [link](https://en.wikipedia.org/w/index.php?title=Leap_year&oldid=1378838087) | CC BY-SA 4.0 |
| `wikipedia_pi.txt` | Pi (Wikipedia) | Wikipedia contributors | revision 1377733505 (2026-09-30T21:27:08Z) | [link](https://en.wikipedia.org/w/index.php?title=Pi&oldid=1377733505) | CC BY-SA 4.0 |
| `wikipedia_python_programming_language.txt` | Python (programming language) (Wikipedia) | Wikipedia contributors | revision 1379554758 (2026-10-10T17:56:06Z) | [link](https://en.wikipedia.org/w/index.php?title=Python_%28programming_language%29&oldid=1379554758) | CC BY-SA 4.0 |
| `wikipedia_burj_khalifa.txt` | Burj Khalifa (Wikipedia) | Wikipedia contributors | revision 1379139570 (2026-10-08T04:42:09Z) | [link](https://en.wikipedia.org/w/index.php?title=Burj_Khalifa&oldid=1379139570) | CC BY-SA 4.0 |
| `wikiquote_narcotics_anonymous.txt` | Narcotics Anonymous (Wikiquote) | Wikiquote contributors | revision 3314544 (2023-06-10T23:04:55Z) | [link](https://en.wikiquote.org/w/index.php?title=Narcotics_Anonymous&oldid=3314544) | CC BY-SA 4.0 |
| `nasa_mars_fact_sheet.txt` | Mars Fact Sheet | Dr. David R. Williams (Author/Curator), NSSDCA, NASA Goddard Space Flight Center | page last updated 19 May 2025 | [link](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) | U.S. Government work (NASA); believed public domain in the U.S. - requires manual confirmation |

All documents were accessed on 2026-10-11. Exact metadata, including content hashes, is
in `collection.json`.

## Important

These documents are *reference material for retrieval*, not ground truth. Wikipedia is a
tertiary, community-edited source. Retrieved passages are candidates that a human must
assess, and they can be outdated or wrong. For example, the "tallest building" claim is
time-sensitive.
