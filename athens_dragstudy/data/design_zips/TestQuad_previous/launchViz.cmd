
FOR /F "skip=2 tokens=2,*" %%A IN ('%SystemRoot%\SysWoW64\REG.exe query "HKLM\software\Metamorph\OpenMETA-Visualizer" /v "PATH"') DO SET DIG_PATH=%%B
"%DIG_PATH%\Dig\run_csvonly.cmd" ".\output.csv"