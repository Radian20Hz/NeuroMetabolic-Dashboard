# Stage C remediation — synthetic acceptance

Raport: [STAGE_C_REMEDIATION_REPORT.md](../../docs/STAGE_C_REMEDIATION_REPORT.md).

`preregistration.json` powstał przed testami; tolerancje i budżet pozostały stałe. `run01_results.json` zachowuje17 PASS/1 FAIL fixture unseen; `run02_results.json` zawiera końcowe19 PASS. `execution_summary.json` wiąże rzeczywiste procesy, final source hashes i synthetic integration hashes.

Runner: `python3 ml/tests/run_stage_c_remediation.py`. Cała dotychczasowa kampania zużyła20/24 forward calls. Nie uruchamiać ponownie pełnego suite w tej kampanii: wymaga10 calls, pozostały4. Nowa kampania wymaga uprzedniej autoryzacji nowego budżetu; nie zerować istniejącego licznika.

Binarne/log artifacts są lokalne, ignorowane: `ml/models/stage_c_remediation_20260914/`. Zero real data, training, backward lub optimizer steps. Trained BEST path i recertyfikacja treningowa pozostają NOT RUN.
