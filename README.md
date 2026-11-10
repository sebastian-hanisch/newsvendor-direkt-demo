# Direkt auf den Kosten lernen: Pinball-Verlust und Quantilregression – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-newsvendor-direkt-demo.streamlit.app/)**

Eigenständige Demo der Reihe **entscheidungsorientiert** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie importiert nichts aus anderen Demos; der Zufallsstrom (SplitMix64) ist aus `entrollte-netze-demo` kopiert und steht im Modulkopf von `nvd_rng.py`. Vehikel ist der **Newsvendor mit Merkmalen** (Bestellmenge, Kosten je fehlende und je überschüssige Einheit); die Frage ist, ob man die Bestellregel **direkt aus den Kosten lernen** sollte, statt erst die Nachfrage zu prognostizieren.

## Warum dieses Problem

Beim Newsvendor ist der Verlust der Entscheidung der **Pinball-Verlust**: Mit den Kosten b je fehlende und h je überschüssige Einheit gilt b·(D − q)₊ + h·(q − D)₊ = (b + h)·ρ_τ(D − q) mit τ = b / (b + h) (getestet, `tests/test_algorithm.py`). Die lineare Bestellregel lässt sich deshalb per **Quantilregression**, ein lineares Programm, direkt auf den Kosten lernen. **Kernfrage:** Lohnt das gegenüber „Prognose plus Reststreuung“? Antwort der Messung: nur unter Bedingungen; in den vier gemessenen Welten waren es diese (Streuung hängt von den Merkmalen ab, die lineare Regel passt, es gibt genug Daten für die Zahl der Merkmale).

## Modell

Nachfrage D = max(0, Y) mit Y ~ Normal(μ(x), σ(x)²), Merkmale x ~ Normal(0, I), h = 1 und b = τ / (1 − τ). Vier Welten:

| Welt | μ(x) | σ(x) |
|---|---|---|
| Wahrheit linear, Streuung konstant (`linear`) | 60 + 10·(x·β) | 8 |
| Streuung wächst linear mit x₁ (`streuung-linear`) | wie oben | max(1, 6 + 4·x₁) |
| Streuung hängt von \|x₁\| ab (`betrag`) | wie oben | 4 + 6·\|x₁\| |
| Wahrheit quadratisch in x₁ (`quadratisch`) | 60 + 10·(x·β) + 8·(x₁² − 1) | 8 |

β ist ein zufälliger Einheitsvektor je Seed. Das Orakel ist das wahre τ-Quantil max(0, μ + σ·z_τ). Die Kosten jeder Regel werden **exakt** als Erwartung über die Nachfrage gerechnet (geschlossene Formel der gestutzten Normalverteilung, Gegenprobe per Monte Carlo im Orakel-Test) und über einen festen Testsatz von 2000 Merkmalspunkten je Seed gemittelt; die Mehrkosten sind der Aufschlag gegenüber dem Orakel in Prozent der Orakelkosten. Bestellmengen unter 0 gelten als 0. Formeln: Expander „📐 Mathematische Formulierung“ der App.

## Methodik

Drei lineare Regeln q(x) = c₀ + x·c, verglichen **im Median über 30 feste Seeds (0 bis 29)**; „direkt besser in k von 30“ zählt Seeds mit strikt weniger Mehrkosten:

* **Mittel bestellen:** Kleinste Quadrate (KQ), der prognostizierte Mittelwert ist die Menge.
* **KQ + Quantil der Residuen** (zweistufig): KQ, dazu das Residuum der Rangzahl ⌈τ·n⌉ als gemeinsamer Aufschlag für alle Merkmalswerte.
* **Direkt:** lineare Quantilregression zum Niveau τ als lineares Programm, `scipy.optimize.linprog` (HiGHS, dünne Matrizen). **Mit L1-Strafe:** Zielfunktion mittlerer Pinball-Verlust plus λ·|c|₁ (ohne Achsenabschnitt); die Stärke steht fest oder wird per **5-facher Kreuzvalidierung** (Faltung = Zeilenindex modulo 5) aus 0 / 0,02 / 0,05 / 0,1 / 0,2 / 0,5 gewählt, bei Gleichstand die kleinere.

Fünf Experimente in der App: (1) Pinball-Verlust von Hand am Handbeispiel, (2) Einzellauf und Bänder („Wann die Streuung zählt“), (3) Raster aus Datenmenge n und Merkmalszahl p, (4) mit Strafe, (5) falsche Klasse. Schnellstart und allgemeine Einstellungen (Welt, τ, n, p, Zufalls-Seed) stehen in der Seitenleiste, die Regler eines einzelnen Experiments (τ im Handbeispiel, Strafe) beim Experiment; der Schnellstart setzt alle gesammelt.

**Handbeispiel** (sechs Beobachtungen, Merkmal x ∈ {0, 1}; bei x = 0 die Nachfragen 4, 5, 6, bei x = 1 die Nachfragen 8, 12, 16; b = 4, h = 1, τ = 0,8), von Hand gerechnet und im Test nachgerechnet: KQ ergibt 5 + 7x (Mengen 5 und 12, Trainingskosten im Mittel 4,167); KQ + Quantil ergibt 6 + 7x (Mengen 6 und 13, 3,500); direkt ergibt 6 + 10x (Mengen 6 und 16, 2,500; Pinball-Summe 3,0, bestätigt durch Aufzählung aller Geraden durch Datenpunktpaare). Das sind **Trainingswerte mit n = 6**, kein Testwert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Mehrkosten gegen das Orakel in Prozent, **Median über die 30 Seeds 0 bis 29**, 2000 Testpunkte je Seed; Quelle `tools/messung_ausgabe.txt` (Abschnitt in Klammern), erzeugt von `tools/messung.py`. „KQ+Q“ steht für KQ + Quantil der Residuen.

| Frage | Befund (Voraussetzung im selben Satz) | Messung und Test |
|---|---|---|
| Mittel bestellen bei schiefen Kosten (A) | Welt linear, p = 3, n = 1000: Mittel bestellen kostet 0,2 % (τ = 0,5), 42,8 % (τ = 0,8) und 287,7 % (τ = 0,95); KQ+Q kostet 0,1 / 0,2 / 0,3 %. Bei n = 100 sind es 44,2 % (τ = 0,8) und 290,9 % (τ = 0,95) und ändern sich zwischen n = 100 und n = 1000 im Median nur wenig (Bänder bei n = 100 breit). | `test_claim_mittel_bestellen_*` (Band) |
| Direkt gewinnt, wenn die Streuung mit x₁ wächst (B) | Welt Streuung linear, p = 3, τ = 0,95: KQ+Q 38,0 % gegen direkt 13,3 % bei n = 100 (direkt besser in 27 von 30 Seeds) und 34,5 % gegen 7,8 % bei n = 1000 (30 von 30); bei n = 5000 33,0 % gegen 7,1 % (30 von 30). Bei τ = 0,8, n = 1000: 13,3 % gegen 1,9 % (30 von 30). | `test_claim_streuung_linear_p3_*` (Band, Zählung ± 3); n = 5000: Messung dieser README, nicht im Test |
| Mit 10 Merkmalen kippt es mit der Datenmenge (C) | Welt Streuung linear, p = 10, τ = 0,95: n = 1000: KQ+Q 33,4 % gegen direkt 9,8 % (30 von 30 Seeds); n = 100: 41,0 % gegen 46,6 % (direkt besser in 13 von 30); n = 30: 54,7 % gegen 185,6 % (0 von 30). | `test_claim_streuung_linear_p10*` (Band, Zählung) |
| Bei konstanter Streuung gewinnt direkt nicht (D) | Welt linear, τ = 0,95: p = 3, n = 1000: KQ+Q 0,3 % gegen direkt 0,9 % (direkt besser in 4 von 30); p = 10, n = 100: 12,7 % gegen 33,2 % (2 von 30); p = 10, n = 30: 47,3 % gegen 197,9 % (1 von 30). Bei τ = 0,8, p = 10, n = 30: 25,7 % gegen 42,7 % (7 von 30). | `test_claim_konstante_streuung_*` (Band, Zählung, Ungleichung) |
| Raster n × p (G) | Welt Streuung linear, τ = 0,95, n ∈ {30, 100, 300, 1000}, p ∈ {1, 3, 10}: direkt hat im Median in 10 von 12 Zellen weniger Mehrkosten als KQ+Q, die zwei Ausnahmen sind p = 10 mit n = 30 (+115,6 Prozentpunkte, direkt besser in 0 von 30) und n = 100 (+1,8 Prozentpunkte, 13 von 30). Welt linear, τ = 0,95: in keiner der 12 Zellen. | `test_claim_raster_*` (Vorzeichenmuster; die Zelle p = 10, n = 100 nur als „nahe der Grenze“) |
| Strafe (E) | Welt linear, p = 10, τ = 0,95, n = 30: ohne Strafe 197,9 %, mit fester Stärke 0,1 63,6 %, mit per Kreuzvalidierung gewählter Stärke 65,7 % (Median der gewählten Stärke 0,10), KQ+Q 47,3 %: die Strafe hilft, erreicht KQ+Q aber nicht. Bei n = 100: ohne Strafe 33,2 %, mit gewählter Stärke 45,3 % (gewählt im Median 0,02), mit fester Stärke 0,1 61,4 %, KQ+Q 12,7 %: Kreuzvalidierung und feste Stärke **verschlechtern**. Welt Streuung linear, n = 30: 185,6 % → 106,1 % (KQ+Q 54,7 %); n = 100: 46,6 % → 46,6 % (KQ+Q 41,0 %). | `test_claim_strafe_p10` (Band ± 20 %, Ungleichungen) |
| Falsche Klasse (F) | p = 3, n = 1000, τ = 0,95: Wahrheit quadratisch in x₁: KQ+Q 141,3 % gegen direkt 143,4 % (direkt besser in 5 von 30); Streuung in \|x₁\|: 19,5 % gegen 20,0 % (7 von 30). Mittel bestellen: 518,8 % und 291,5 %. Der Unterschied zwischen den beiden Regeln beträgt in beiden Welten höchstens ein Zehntel des kleineren Werts. | `test_claim_falsche_klasse` (Band, Ungleichung) |
| Direkt hat auf den Trainingsdaten nie einen höheren Pinball-Verlust | In 40 Zufallsinstanzen (n 6 bis 39, p 1 bis 4, τ 0,5 / 0,8 / 0,95) ist der Trainings-Pinball-Verlust der direkten Regel nie größer als der von KQ+Q oder Mittel bestellen (beide in derselben linearen Klasse; ohne Abschneiden der Mengen bei 0). | `test_direct_never_costs_more_than_two_stage_on_the_training_data` |

Zusätze, **nicht im Test**: Raster der Welt Streuung linear, p = 10, n = 100 liegt knapp (Vorzeichen kann je Plattform wechseln); Abschnitt H der Messung (siehe unten); alle Zellen der Messung, die hier nicht stehen.

### Befunde und Korrekturen gegenüber dem Plan

Der „Plan“ ist die Vormessung vor dem Bau (`eo_v2_news.py`, `eo_v2b_reg.py` der Planung: 30 Seeds, 20 000 Test-Ziehungen der Nachfrage je Seed, Zufallsstrom von numpy, Quantil der Residuen linear interpoliert). Im Bau gilt die neue Messung; die Aussagen gelten qualitativ alle, einzelne Zahlen und **eine Zelle** weichen ab:

| Zelle (Median Mehrkosten in %, KQ+Q gegen direkt) | Plan | Bau |
|---|---|---|
| Streuung linear, p = 3, n = 100, τ = 0,95 | 38,5 gegen 13,8 (29 von 30) | 38,0 gegen 13,3 (27 von 30) |
| Streuung linear, p = 3, n = 1000, τ = 0,95 | 33,9 gegen 8,1 (30 von 30) | 34,5 gegen 7,8 (30 von 30) |
| Welt linear, p = 3, n = 1000, τ = 0,95 | 0,3 gegen 0,8 | 0,3 gegen 0,9 |
| Welt linear, p = 10, n = 30, τ = 0,95 | 64,2 gegen 221,3 (0 von 30) | 47,3 gegen 197,9 (1 von 30) |
| Streuung linear, p = 10, n = 30, τ = 0,95 | 104,8 gegen 208,0 | 54,7 gegen 185,6 |
| **Streuung linear, p = 10, n = 100, τ = 0,95** | **42,1 gegen 34,1 (direkt besser in 19 von 30)** | **41,0 gegen 46,6 (13 von 30): das Vorzeichen kehrt sich um** |
| Falsche Klasse quadratisch / \|x₁\|, n = 1000, τ = 0,95 | 138,5 gegen 141,6 / 19,5 gegen 20,9 | 141,3 gegen 143,4 / 19,5 gegen 20,0 |
| Strafe, Welt linear, n = 30, τ = 0,95: direkt → mit Kreuzvalidierung | 221,3 → 67,7 (KQ+Q 64,2) | 197,9 → 65,7 (KQ+Q 47,3) |
| Strafe, Welt linear, n = 100, τ = 0,95: ohne → mit Kreuzvalidierung | 32,1 → 39,9 | 33,2 → 45,3 |

* **Die Zelle p = 10, n = 100 in der Welt „Streuung linear“** ist die einzige, in der sich eine Aussage der Vormessung („direkt besser“) nicht reproduziert: Sie liegt nahe der Grenze (im Raster der Messung G: +1,8 Prozentpunkte Median der gepaarten Differenzen, 13 von 30 Seeds); die README sagt deshalb nur, dass die Vorzeichen bei p = 10 von der Datenmenge abhängen (kleines n: KQ+Q besser, n = 1000: direkt besser).
* **Ursachen der Abweichungen, soweit gemessen:** andere Zufallsziehungen (SplitMix64 statt numpy), exakte statt simulierte Testkosten, Rangregel statt linear interpoliertes Quantil. Nur der Anteil der Rangregel ist getrennt gemessen (Abschnitt H): bei Welt linear, p = 10, n = 30, τ = 0,95 ergibt KQ+Q mit der Rangregel 47,3 %, mit dem linear interpolierten Quantil 55,3 % (Plan: 64,2 %); in den drei übrigen gemessenen Zellen (Welt linear mit p = 10 und n = 100, Streuung linear mit p = 3 und n = 100, Welt linear mit p = 3 und n = 1000) weichen beide Fassungen um höchstens 0,2 Prozentpunkte voneinander ab. Der Rest der Abweichung lässt sich den beiden anderen Ursachen nicht einzeln zuordnen.
* **Neu gegenüber dem Plan:** Raster n × p (G), die Zählung „besser in k von 30“ je Zelle, die feste Strafstärke als Regler. Eine feste Stärke von 0,1 ist bei n = 100 (Welt linear, p = 10, τ = 0,95) mit 61,4 % klar schlechter als keine Strafe (33,2 %): die passende Stärke hängt hier von der Datenmenge ab (feste Stärke 0,1: 63,6 % bei n = 30 gegen 197,9 % ohne Strafe, aber 61,4 % bei n = 100 gegen 33,2 % ohne Strafe); wie sie von der Skala der Nachfrage abhängt, ist nicht gemessen.

Die Befunde bleiben Messungen **dieser** Welten (Normalrauschen, bei 0 gestutzt) und dieser Regeln; sie gelten nicht für „die“ Quantilregression oder „die“ zweistufige Methode.

## Ehrliche Grenzen

* **Simulierte Daten, eine Familie:** Normalrauschen mit Stutzung bei 0, vier Welten, 30 Seeds, ein Testsatz von 2000 Merkmalspunkten je Seed. Keine Echtdaten; Zahlen aus Echtdaten-Experimenten der Literatur werden **nicht** zitiert.
* **Die zweistufige Vergleichsregel ist nur KQ + Quantil der Residuen** mit gemeinsamem Aufschlag. Eine zweistufige Regel mit modellierter Streuung (etwa eine zweite Regression auf die Residuen) wurde nicht gemessen; „direkt gewinnt“ gilt gegenüber der gemessenen Regel, nicht gegenüber jeder zweistufigen Methode. In der Vormessung gab es zusätzlich KQ + Normalquantil der Reststreuung; die App zeigt es nicht.
* **Das Orakel kennt die wahre Verteilung.** Mehrkosten gegen das Orakel sind in der Praxis nicht messbar; dort bleibt nur der Vergleich auf Testdaten.
* **Strafe:** nur L1, nur die 5-fache Kreuzvalidierung als Wahlregel, ein Raster mit sechs Stufen; eine andere Wahlregel oder L2-Strafe wurde nicht geprüft. Dass die Kreuzvalidierung bei n = 100 verschlechtert, ist in dieser Welt und mit dieser Regel gemessen; die Deutung „die Wahl der Strafe ist selbst verrauscht“ ist **nicht gemessen** (die App kennzeichnet sie so).
* **Lineare Klasse:** Die Welten „quadratisch“ und „|x₁|“ zeigen nur, dass der Verlust allein die falsche Klasse in diesen Welten nicht repariert; ob eine flexiblere Klasse es tut, wird hier nicht gemessen.
* **Quelle, nicht im Volltext gelesen:** Ban, G.-Y. und Rudin, C., *The Big Data Newsvendor: Practical Insights from Machine Learning*, Operations Research 67(1), 2019. Laut veröffentlichtem Abstract lernen sie die Bestellmenge direkt aus Merkmalen und Nachfragen statt über eine Prognose (Volltext, Satzbedingungen, Zahlen und Seitenangaben **nicht gelesen**). Weitere Inhalte, die in früheren Fassungen dieser Demo standen (ERM mit und ohne Regularisierung, hohe Dimension, inkonsistente Entscheidungen beim Weglassen von Merkmalen), stammen aus der Zusammenfassung einer Websuche; der Abstract im Wortlaut ist dazu nicht geprüft. Nichts davon wird hier als Befund verwendet; alle Zahlen stammen aus den Messungen dieser README.
* **Rechenzeit:** Alle Experimente rechnen live (je Seed ein lineares Programm, bei der Kreuzvalidierung 30). Die Dauer hängt vom Rechner ab und ist nicht festgeschrieben; Ergebnisse werden je Einstellung zwischengespeichert.

### Wo die Annahmen enden

(Dieselbe Tabelle wie in der App; ein Test vergleicht beide.)

| Annahme | Was passiert, wenn sie verletzt ist (30 Seeds, Median der Mehrkosten gegen das Orakel, τ = 0,95) | Ausweg (soweit gemessen) |
|---|---|---|
| **Die Streuung hängt von den Merkmalen ab** | Bei konstanter Streuung (Welt „Wahrheit linear, Streuung konstant“, p = 3, n = 1000) kostet direkt 0,9 % gegen 0,3 % für KQ plus Quantil; direkt ist in 4 von 30 Seeds besser. | Zweistufig bleibt die einfache Regel. |
| **Genug Daten für die Zahl der Merkmale** | Mit p = 10 und n = 30 (Welt linear) kostet direkt 197,9 % gegen 47,3 % für KQ plus Quantil; direkt ist in 1 von 30 Seeds besser. | L1-Strafe holt einen Teil: 65,7 % mit per Kreuzvalidierung gewählter Strafe, KQ plus Quantil erreicht sie nicht. |
| **Die Wahl der Strafe ist verlässlich** | Mit p = 10 und n = 100 (Welt linear) steigt der Median von 33,2 % ohne Strafe auf 45,3 % mit per Kreuzvalidierung gewählter Strafe. | Nicht gemessen; eine andere Wahlregel wurde nicht geprüft. |
| **Die lineare Klasse trifft die Wahrheit** | Bei quadratischem Mittelwert (p = 3, n = 1000) 143,4 % gegen 141,3 %; bei Streuung, die vom Betrag von x₁ abhängt, 20,0 % gegen 19,5 %. In diesen beiden Welten repariert der Verlust allein die Klasse nicht. | Nicht gemessen; eine flexiblere Klasse ist hier nicht umgesetzt. |
| **Eine Familie von Instanzen** | Nur Normalrauschen, bei 0 gestutzt, vier Welten, simulierte Daten; keine Echtdaten, keine anderen Verteilungen. | Eigene Daten prüfen. |

## Verifikation

127 Tests (`pytest --collect-only -q tests`), alle grün.

* **Kern gegen Handrechnung:** das Handbeispiel (KQ 5 + 7x, KQ + Quantil 6 + 7x, direkt 6 + 10x, Trainingskosten 4,167 / 3,500 / 2,500, Pinball-Summe 3,0), Identität Kosten = (b + h)·Pinball, Rangregel des Quantils (`tests/test_algorithm.py`); Handwerte der Welten und des wahren Quantils (`tests/test_scenario.py`).
* **Orakel-Tests** (`tests/test_oracle_newsvendor_direkt.py`, anderer Rechenweg): das LP gegen **Aufzählung aller Ecken** (Geraden durch Datenpunktpaare, p = 1; mit und ohne Strafe, 60 Instanzen, auch mit vielen gleichen x-Werten), gegen **`sklearn.linear_model.QuantileRegressor`** (nur Test, `pytest.importorskip`), erwartete Kosten gegen **Monte Carlo** (inklusive starker Stutzung bei 0), wahres Quantil gegen das empirische Quantil von Ziehungen, Optimalität des Orakels, SplitMix64 gegen eine zweite Ganzzahl-Implementierung und den veröffentlichten Referenzwert für Seed 0. Im Scratch-Lauf (nicht im Repo) wurden je 250 Instanzen gegen die Aufzählung (n 3 bis 13, τ 0,1 / 0,5 / 0,8 / 0,95, mit und ohne Strafe) und gegen scikit-learn (n 12 bis 79, p 1 bis 6) gerechnet; die größte Abweichung im Zielwert war 1,5·10⁻¹⁴. Im Repo steht die kleine Fassung.
* **Eigenschaften:** Trainings-Pinball nimmt mit steigender Strafe nicht ab, die L1-Norm nicht zu; Verschieben aller Nachfragen um c verschiebt nur den Achsenabschnitt; Skalieren skaliert alle Koeffizienten; Vertauschen der Beobachtungen ändert nichts; τ = 0,5 ist die Median-Regression; direkt hat auf den Trainingsdaten nie einen höheren Pinball-Verlust als KQ+Q (ohne Abschneiden der Mengen bei 0).
* **Randfälle** (`tests/test_edge_cases.py`): n < p + 1 (das LP passt exakt, die Strafe verhindert es), n = p + 1, gleiche x-Werte, gleiche Nachfragen, τ = 0,95 mit n = 10 < 20, n = 1, ungültige Eingaben.
* **Verdikte** (`tests/test_verdicts.py`): jede Schwelle (zwei Drittel / ein Drittel der Seeds, ein Zehntel, Hälfte der Zellen) und jede Zahl im Text an handgesetzten Werten.
* **Zahlen der README:** `tests/test_claims.py` (Bänder an den festen Seeds, siehe Tabelle der Befunde); die Tabelle „Wo die Annahmen enden“ steht in App und README wortgleich (ein Test vergleicht).
* **Oberfläche und Layout:** AppTest-Rauchtest (jeder Button, jeder Regler auf allen Stufen, Permalink mit Einrasten, ungültige Werte), Presets, aktives Preset hervorgehoben, jeder Regler steht in `SETTING_SPECS` (`tests/test_layout.py`), Footer-Test, Laufzeit-Importe (`tests/test_runtime_imports.py`).
* **Nicht geprüft:** das Aussehen im Browser bei hellem und dunklem Design, mit echtem Mausklick; die Messung mit n = 5000 und Abschnitt H laufen nur in `tools/messung.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Handbeispiel, Bänder, Einzellauf, Raster, Strafe, falsche Klasse, Grenzen, Formeln |
| `nvd_algorithm.py` | Kern: KQ, Quantil der Residuen, Quantilregression (LP, HiGHS), Kreuzvalidierung, Pinball und Newsvendor-Kosten |
| `nvd_constants.py` | Regler-Stufen, Welten, Farben, Presets, Handbeispiel |
| `nvd_evaluation.py` | Bänder über feste Seeds, Raster, Strafe-Messung, Verdikte als reine Funktionen |
| `nvd_layout.py` | Fester Kopf (Stylesheet) |
| `nvd_pdf_export.py` | PDF der aktuellen Einstellung (fpdf2) |
| `nvd_presets.py` | Permalink (Einrasten und Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `nvd_rng.py` | SplitMix64 (Kopie aus `entrollte-netze-demo`, gekürzt) |
| `nvd_scenario.py` | Welten, Daten, wahres Quantil, exakte erwartete Kosten |
| `nvd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur Orakel-Gegenprobe) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, SciPy, Plotly, fpdf2 |
| `tools/messung.py`, `tools/messung_ausgabe.txt` | Messung hinter den Zahlen der README und ihre Ausgabe |
| `tests/` | Handrechnung, Orakel, Eigenschaften, Randfälle, Verdikte, Zahlen der README, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

* **Gewichtete Stichproben** (nächste Nachbarn, Kerngewichte, Zufallswald) als nichtparametrische Alternative: hier nicht Thema; scikit-learn ist kein Laufzeitpaket.
* **Echtdaten** und Zahlen aus Echtdaten-Experimenten der Literatur.
* **L2-Strafe**, andere Wahlregeln für die Strafe, Strafe auf einer flexibleren Klasse.
* **Beweise** und Raten für endliche Stichproben: die Demo misst, sie beweist nichts.
* **Feste Rechenzeiten** in App oder README.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (HiGHS).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html).
