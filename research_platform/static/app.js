const translations = {
  zh: {
    subtitle: "轻量视觉-语言-动作研究平台", connecting: "正在连接本地服务", connected: "本地服务在线", disconnected: "本地服务离线",
    navLive: "实时仿真", navBenchmark: "批量评测", navGovernance: "研究治理", navPortfolio: "研究地图", navReleases: "研究发布", navTraining: "训练监控", navExperiments: "实验台账", navEvidence: "研究证据", navLegacy: "原展示页",
    liveEyebrow: "实时仿真会话", liveTitle: "从命令到机械臂动作", readyTitle: "MuJoCo 已就绪", readyBody: "输入指令或从右侧选择任务开始真实仿真。",
    phase: "阶段", elapsed: "用时", targetError: "目标误差", objectHeight: "物体高度", contacts: "接触数", result: "结果",
    commandLabel: "自然语言指令", commandPlaceholder: "把蓝色方块放到蓝色目标区", execute: "执行", hintBlueBlue: "蓝块 → 蓝区", hintLeftmost: "最左方块 → 白碗",
    sessionConfig: "会话配置", task: "任务", method: "方法", policyRgb: "RGB 几何 + 结构化执行", policyState: "MuJoCo 状态参考专家",
    layoutComplexity: "场景复杂度", speed: "速度", startSimulation: "启动实时仿真", pause: "暂停", resume: "继续", reset: "重置", stop: "停止",
    openNativeViewer: "打开原生 MuJoCo Viewer", runtimeBoundary: "运行时位置来源",
    trainingEyebrow: "真实训练进程", trainingTitle: "观察损失，而不是播放动画", trainingIntro: "每个点都来自仓库中实际训练脚本的标准输出。",
    lossCurve: "归一化损失曲线", trainingConfig: "训练配置", dataset: "数据集", startTraining: "开始训练", stopTraining: "停止训练", modelOutput: "模型输出",
    evidenceEyebrow: "已验证研究结果", evidenceTitle: "结果、边界与可复现入口", openSummary: "打开双语研究摘要", retainedResult: "最终方案严格成功",
    semanticSelection: "语义与初始选物正确", firstAttempt: "首尝试成功", rejected: "未保留", scopeTitle: "结论边界",
    scopeBody: "MuJoCo-only。无真实 WidowX、无 Isaac、无端到端 OpenVLA/RT-2 微调。",
    legacyEyebrow: "原始展示资产", legacyTitle: "Integrated Research Showcase", legacyNotice: "以下页面按原文件加载，未修改内容。",
    success: "成功", failure: "失败", running: "运行中", paused: "已暂停", stopped: "已停止", completed: "已完成", failed: "发生错误", interrupted: "意外中断", idle: "等待",
    commandUnknown: "无法识别该指令。请使用颜色与目标，或输入 pause / resume / stop / reset。", requestFailed: "请求失败",
    viewerStarted: "已启动原生 MuJoCo Viewer。首次运行 CLIP 方案可能需要加载模型。", trainingStarted: "训练进程已启动。",
    benchmarkEyebrow: "闭环批量协议", benchmarkTitle: "跨任务、跨种子的真实基准评测", benchmarkIntro: "每个 episode 都运行真实 MuJoCo，并自动归档结果与复现配置。",
    currentEpisode: "当前 episode", episodes: "Episodes", successRate: "成功率", meanError: "平均目标误差", episodeResults: "逐 episode 结果", noBenchmarkResults: "尚无评测结果",
    policyRates: "视觉 / 状态成功率", pairDisagreements: "配对分歧", pairedEvidence: "配对统计证据", pairedMethods: "同 Seed 配对方法", ci95: "95% 置信区间",
    bothSuccess: "双方成功", rgbOnly: "仅视觉成功", stateOnly: "仅状态成功", bothFail: "双方失败", noPairedEvidence: "完成双方法 episode 后生成配对统计。",
    benchmarkConfig: "评测配置", taskCoverage: "任务覆盖", taskBlueBlue: "蓝块 → 蓝区", taskBlueRed: "蓝块 → 红区", taskRedRed: "红块 → 红区", taskLeftmost: "最左方块 → 白碗",
    seedStart: "起始 Seed", seedsPerTask: "每任务数量", startBenchmark: "开始批量评测", stopBenchmark: "停止评测", protocolRule: "协议规则", protocolRuleBody: "相同任务与 seed 依次运行两种方法；最多 40 个 episode；报告 95% 置信区间与配对分歧。",
    experimentsEyebrow: "可追溯研究记录", experimentsTitle: "实验台账与方法比较", experimentsIntro: "平台运行自动形成追加式记录，不覆盖已有数据。", exportCsv: "导出 CSV",
    allRuns: "全部运行", simulationRuns: "仿真", trainingRuns: "训练", benchmarkRuns: "批量评测", policyComparison: "策略闭环汇总", platformRunsOnly: "仅平台新运行",
    recentRuns: "最近运行", allTypes: "全部类型", allStatuses: "全部状态", type: "类型", configuration: "配置", status: "状态", keyMetric: "关键指标", started: "开始时间", noRuns: "尚无平台运行记录",
    runInspector: "运行详情", artifact: "产物", reproductionCommand: "复现命令", copyCommand: "复制命令", systemHealth: "系统健康", commandCopied: "复现命令已复制。",
    downloadJson: "下载 JSON 报告", downloadMarkdown: "下载 Markdown", visualInput: "初始俯视场景", finalState: "最终前视状态", groundingError: "视觉定位误差",
    governanceTitle: "把实验变成可审计的研究决策", governanceIntro: "先锁定问题与门槛，再运行 MuJoCo；负结果同样进入证据链。", registeredStudies: "已注册研究",
    traceTarget: "Target 问题", traceRegister: "Register 协议", traceAcquire: "Acquire 证据", traceCheck: "Check 门控", traceExplain: "Explain 决策",
    protocolBuilder: "协议预注册", studyTitle: "研究标题", hypothesis: "研究假设", successFloor: "成功率下限 %", ciWidth: "置信区间宽度 %", targetCeiling: "目标误差上限 mm", groundingCeiling: "定位误差上限 mm",
    plannedWorkload: "计划工作量", lockProtocol: "锁定协议与指纹", protocolRegistry: "协议注册表", appendOnly: "只追加，不覆盖", noStudies: "尚无预注册研究。",
    selectStudy: "选择一个研究协议", selectStudyBody: "系统将展示协议指纹、执行状态和证据门控。", launchProtocol: "按协议运行", rerunProtocol: "按协议复跑", decisionMemo: "决策备忘录", evidenceGates: "证据门控",
    verdictLocked: "协议已锁定", verdictExecuting: "正在取证", verdictReady: "可写入报告", verdictNeedsEvidence: "证据不足", verdictFailed: "执行失败", protocolCreated: "协议已锁定，后续配置不可修改。", protocolLaunched: "已按锁定协议启动真实配对评测。",
    gateProtocol: "协议指纹已锁定", gateExecution: "预期 episode 已完成", gatePairing: "同 seed 配对完整", gateArtifacts: "视觉证据已归档", gateSuccess: "方法成功率达到下限", gateTarget: "目标误差低于上限", gateGrounding: "RGB 定位误差低于上限", gateUncertainty: "置信区间宽度合格",
    gatePass: "通过", gateFail: "未通过", gatePending: "待验证", expectedEpisodes: "预期 Episodes", latestBenchmark: "最新 Benchmark", completedPairs: "完整配对", passedGates: "通过门控",
    portfolioTitle: "从实验路线到可写论文主张", portfolioIntro: "每条结论绑定指标、证据指纹与禁止越界表述；不同协议不混成排行榜。", portfolioClaims: "论文主张", portfolioMethods: "方法节点", portfolioIntegrity: "完整性门控", portfolioExport: "导出组合报告",
    portfolioFlowClaim: "提出主张", portfolioFlowEvidence: "绑定证据", portfolioFlowBoundary: "锁定边界", portfolioFlowWrite: "决定写法", claimRegister: "主张注册表", claimRegisterHint: "点击检查写作就绪度", writingReadiness: "写作就绪度",
    permittedWording: "允许写入", prohibitedWording: "禁止越界", claimGates: "主张门控", evidenceSources: "证据来源", nextDecision: "下一决策", openTraceStudy: "查看 TRACE 协议", methodLifecycle: "方法生命周期", notLeaderboard: "非排行榜", sourceIntegrity: "来源完整性",
    claimReportable: "可写入", claimBounded: "限定表述", claimNegative: "负结果", claimBlocked: "范围外", lifecycleReference: "参考专家", lifecycleControl: "控制 / 数据", lifecycleBaseline: "学习对照", lifecycleProbe: "轻量探针", lifecycleRetained: "保留方案", lifecycleRejected: "淘汰候选", noMethods: "该阶段暂无方法",
    releaseTitle: "把论文证据冻结为可核验版本", releaseIntro: "发布包保存来源副本、主张状态、TRACE 指纹和实验台账；历史版本只追加、不覆盖。", releaseReadiness: "发布状态", releaseCount: "已冻结版本", bundleFiles: "每包文件", releaseBuilder: "证据版本发布", releaseLabel: "版本标签", releaseNote: "发布说明", currentPortfolioDigest: "当前 Portfolio 指纹", createRelease: "冻结新证据版本", releaseRule: "发布规则", releaseRuleBody: "仅在五项门控全部通过且没有任务运行时发布。每次发布生成新 ID，不提供修改或删除入口。",
    selectRelease: "选择一个证据版本", selectReleaseBody: "查看冻结主张、文件校验结果及相对当前工作区的漂移。", downloadManifest: "下载 JSON 清单", downloadReleaseReadme: "下载双语说明", frozenClaims: "冻结主张", bundleVerification: "文件级校验", releaseHistory: "发布历史", noReleases: "尚无证据版本。", releaseReady: "可以发布", releaseBlocked: "门控未通过", releaseVerifiedCurrent: "当前一致", releaseVerifiedSnapshot: "历史快照有效", releaseCorrupted: "校验失败", releaseCreated: "证据版本已冻结，清单和文件副本已归档。", currentWorkspace: "当前工作区", frozenSnapshot: "冻结快照", manifestIntegrity: "清单完整", fileIntegrity: "文件完整", portfolioMatch: "Portfolio 一致", ledgerMatch: "台账一致", releaseGatePortfolio: "来源完整性", releaseGateClaim: "可写主张", releaseGateTrace: "TRACE 纳管", releaseGateLedger: "台账可用", releaseGateIdle: "任务静止",
  },
  en: {
    subtitle: "Lightweight vision-language-action research platform", connecting: "Connecting to local service", connected: "Local service online", disconnected: "Local service offline",
    navLive: "Live simulation", navBenchmark: "Benchmark lab", navGovernance: "Research governance", navPortfolio: "Research map", navReleases: "Evidence releases", navTraining: "Training monitor", navExperiments: "Experiment ledger", navEvidence: "Research evidence", navLegacy: "Original showcase",
    liveEyebrow: "Live simulation session", liveTitle: "From command to robot motion", readyTitle: "MuJoCo is ready", readyBody: "Enter a command or select a task to start a real simulation.",
    phase: "Phase", elapsed: "Elapsed", targetError: "Target error", objectHeight: "Object height", contacts: "Contacts", result: "Result",
    commandLabel: "Natural-language command", commandPlaceholder: "Place the blue cube on the blue target", execute: "Run", hintBlueBlue: "Blue cube → blue", hintLeftmost: "Leftmost cube → bowl",
    sessionConfig: "Session configuration", task: "Task", method: "Method", policyRgb: "RGB geometry + structured execution", policyState: "MuJoCo state-reference expert",
    layoutComplexity: "Scene complexity", speed: "Speed", startSimulation: "Start live simulation", pause: "Pause", resume: "Resume", reset: "Reset", stop: "Stop",
    openNativeViewer: "Open native MuJoCo Viewer", runtimeBoundary: "Runtime position source",
    trainingEyebrow: "Real training process", trainingTitle: "Inspect loss, not an animation", trainingIntro: "Every point comes from the actual training scripts in the repository.",
    lossCurve: "Normalised loss curve", trainingConfig: "Training configuration", dataset: "Dataset", startTraining: "Start training", stopTraining: "Stop training", modelOutput: "Model output",
    evidenceEyebrow: "Validated research result", evidenceTitle: "Results, boundaries and reproduction", openSummary: "Open bilingual research summary", retainedResult: "Retained strict success",
    semanticSelection: "Semantic + initial selection correct", firstAttempt: "First-attempt success", rejected: "Not retained", scopeTitle: "Claim boundary",
    scopeBody: "MuJoCo only. No physical WidowX, no Isaac and no end-to-end OpenVLA/RT-2 fine-tuning.",
    legacyEyebrow: "Original showcase asset", legacyTitle: "Integrated Research Showcase", legacyNotice: "The page below is loaded from the original file without modification.",
    success: "SUCCESS", failure: "FAILURE", running: "RUNNING", paused: "PAUSED", stopped: "STOPPED", completed: "COMPLETED", failed: "ERROR", interrupted: "INTERRUPTED", idle: "IDLE",
    commandUnknown: "Command not recognised. Include an object colour and target, or use pause / resume / stop / reset.", requestFailed: "Request failed",
    viewerStarted: "Native MuJoCo Viewer started. The CLIP route may load the model on first use.", trainingStarted: "Training process started.",
    benchmarkEyebrow: "Closed-loop batch protocol", benchmarkTitle: "Real benchmarks across tasks and seeds", benchmarkIntro: "Every episode runs in MuJoCo and archives its result and reproduction configuration.",
    currentEpisode: "Current episode", episodes: "Episodes", successRate: "Success rate", meanError: "Mean target error", episodeResults: "Per-episode results", noBenchmarkResults: "No benchmark results yet",
    policyRates: "RGB / state success", pairDisagreements: "Paired disagreements", pairedEvidence: "Paired statistical evidence", pairedMethods: "Same-seed paired methods", ci95: "95% confidence interval",
    bothSuccess: "Both succeed", rgbOnly: "RGB only", stateOnly: "State only", bothFail: "Both fail", noPairedEvidence: "Paired statistics appear after both methods complete.",
    benchmarkConfig: "Benchmark configuration", taskCoverage: "Task coverage", taskBlueBlue: "Blue cube → blue", taskBlueRed: "Blue cube → red", taskRedRed: "Red cube → red", taskLeftmost: "Leftmost cube → bowl",
    seedStart: "Starting seed", seedsPerTask: "Runs per task", startBenchmark: "Start benchmark", stopBenchmark: "Stop benchmark", protocolRule: "Protocol rule", protocolRuleBody: "Both methods run on each task and seed; at most 40 episodes; report 95% intervals and paired disagreements.",
    experimentsEyebrow: "Traceable research records", experimentsTitle: "Experiment ledger and method comparison", experimentsIntro: "Platform runs create append-only records without overwriting existing data.", exportCsv: "Export CSV",
    allRuns: "All runs", simulationRuns: "Simulation", trainingRuns: "Training", benchmarkRuns: "Benchmark", policyComparison: "Closed-loop policy summary", platformRunsOnly: "New platform runs only",
    recentRuns: "Recent runs", allTypes: "All types", allStatuses: "All statuses", type: "Type", configuration: "Configuration", status: "Status", keyMetric: "Key metric", started: "Started", noRuns: "No platform runs yet",
    runInspector: "Run inspector", artifact: "Artifact", reproductionCommand: "Reproduction command", copyCommand: "Copy command", systemHealth: "System health", commandCopied: "Reproduction command copied.",
    downloadJson: "Download JSON report", downloadMarkdown: "Download Markdown", visualInput: "Initial top-view scene", finalState: "Final front-view state", groundingError: "RGB localisation error",
    governanceTitle: "Turn experiments into auditable decisions", governanceIntro: "Lock the question and thresholds before MuJoCo execution; negative results stay in the evidence chain.", registeredStudies: "Registered studies",
    traceTarget: "Target question", traceRegister: "Register protocol", traceAcquire: "Acquire evidence", traceCheck: "Check gates", traceExplain: "Explain decision",
    protocolBuilder: "Protocol pre-registration", studyTitle: "Study title", hypothesis: "Hypothesis", successFloor: "Success floor %", ciWidth: "CI width %", targetCeiling: "Target error ceiling mm", groundingCeiling: "Grounding ceiling mm",
    plannedWorkload: "Planned workload", lockProtocol: "Lock protocol + fingerprint", protocolRegistry: "Protocol registry", appendOnly: "Append-only", noStudies: "No pre-registered studies yet.",
    selectStudy: "Select a study protocol", selectStudyBody: "The protocol fingerprint, execution state and evidence gates will appear here.", launchProtocol: "Run locked protocol", rerunProtocol: "Repeat locked protocol", decisionMemo: "Decision memo", evidenceGates: "Evidence gates",
    verdictLocked: "Protocol locked", verdictExecuting: "Acquiring evidence", verdictReady: "Ready to report", verdictNeedsEvidence: "More evidence needed", verdictFailed: "Execution failed", protocolCreated: "Protocol locked. Its configuration is now immutable.", protocolLaunched: "Real paired evaluation launched from the locked protocol.",
    gateProtocol: "Protocol fingerprint locked", gateExecution: "Expected episodes completed", gatePairing: "Same-seed pairs complete", gateArtifacts: "Visual evidence archived", gateSuccess: "Policy success floor met", gateTarget: "Target error below ceiling", gateGrounding: "RGB grounding below ceiling", gateUncertainty: "Confidence interval width acceptable",
    gatePass: "PASS", gateFail: "FAIL", gatePending: "PENDING", expectedEpisodes: "Expected episodes", latestBenchmark: "Latest benchmark", completedPairs: "Complete pairs", passedGates: "Passed gates",
    portfolioTitle: "From experimental route to defensible thesis claims", portfolioIntro: "Every claim binds metrics, source fingerprints and prohibited overclaims; incompatible protocols are never pooled into a leaderboard.", portfolioClaims: "Thesis claims", portfolioMethods: "Method nodes", portfolioIntegrity: "Integrity gates", portfolioExport: "Export portfolio",
    portfolioFlowClaim: "State claim", portfolioFlowEvidence: "Bind evidence", portfolioFlowBoundary: "Lock boundary", portfolioFlowWrite: "Decide wording", claimRegister: "Claim register", claimRegisterHint: "Select to inspect writing readiness", writingReadiness: "Writing readiness",
    permittedWording: "Permitted wording", prohibitedWording: "Prohibited overclaim", claimGates: "Claim gates", evidenceSources: "Evidence sources", nextDecision: "Next decision", openTraceStudy: "Open TRACE protocol", methodLifecycle: "Method lifecycle", notLeaderboard: "Not a leaderboard", sourceIntegrity: "Source integrity",
    claimReportable: "REPORTABLE", claimBounded: "BOUNDED", claimNegative: "NEGATIVE", claimBlocked: "OUT OF SCOPE", lifecycleReference: "Reference", lifecycleControl: "Control / data", lifecycleBaseline: "Learned baseline", lifecycleProbe: "Lightweight probe", lifecycleRetained: "Retained", lifecycleRejected: "Rejected", noMethods: "No methods in this stage",
    releaseTitle: "Freeze thesis evidence into a verifiable release", releaseIntro: "Each release bundles source copies, claim states, TRACE fingerprints and the experiment ledger; historical releases are append-only.", releaseReadiness: "Release readiness", releaseCount: "Frozen releases", bundleFiles: "Files per bundle", releaseBuilder: "Evidence release", releaseLabel: "Release label", releaseNote: "Release note", currentPortfolioDigest: "Current portfolio digest", createRelease: "Freeze new evidence release", releaseRule: "Release rule", releaseRuleBody: "All five gates must pass and no job may be active. Every release receives a new ID; the platform exposes no edit or delete action.",
    selectRelease: "Select an evidence release", selectReleaseBody: "Inspect frozen claims, file-level verification and drift relative to the current workspace.", downloadManifest: "Download JSON manifest", downloadReleaseReadme: "Download bilingual note", frozenClaims: "Frozen claims", bundleVerification: "File verification", releaseHistory: "Release history", noReleases: "No evidence releases yet.", releaseReady: "READY TO RELEASE", releaseBlocked: "GATES BLOCKED", releaseVerifiedCurrent: "VERIFIED CURRENT", releaseVerifiedSnapshot: "VERIFIED SNAPSHOT", releaseCorrupted: "CORRUPTED", releaseCreated: "Evidence release frozen with its manifest and bundled source copies.", currentWorkspace: "Current workspace", frozenSnapshot: "Frozen snapshot", manifestIntegrity: "Manifest integrity", fileIntegrity: "File integrity", portfolioMatch: "Portfolio match", ledgerMatch: "Ledger match", releaseGatePortfolio: "Source integrity", releaseGateClaim: "Reportable claim", releaseGateTrace: "TRACE capture", releaseGateLedger: "Ledger available", releaseGateIdle: "Jobs quiescent",
    defaultStudyTitle: "Paired comparison of RGB grounding and the state-reference expert",
    defaultStudyHypothesis: "On identical tasks and seeds, the RGB-grounded policy can approach the state-reference success rate while keeping mean localisation error below the registered threshold.",
    switchLanguage: "Switch to Chinese",
    localLanguageStudy: "Study recorded in another interface language",
    localLanguageHypothesis: "Switch to the Chinese interface to view the original study wording.",
    localLanguageRelease: "Release recorded in another interface language",
  }
};

Object.assign(translations.zh, {
  defaultStudyTitle: "RGB 定位策略与状态参考专家的配对比较",
  defaultStudyHypothesis: "在相同任务和 seed 下，RGB 定位策略能够保持接近状态参考专家的成功率，同时将平均定位误差控制在预设范围内。",
  switchLanguage: "切换到英文",
  localLanguageStudy: "使用其他界面语言记录的研究",
  localLanguageHypothesis: "切换到英文界面查看对应内容。",
  localLanguageRelease: "使用其他界面语言记录的版本",
});

Object.assign(translations.zh, {
  navTraining: "\u4f4e\u8d44\u6e90\u9002\u914d",
  trainingRuns: "\u8bad\u7ec3 / \u9002\u914d", adaptationRuns: "\u4f4e\u8d44\u6e90\u9002\u914d",
  trainingEyebrow: "\u4f4e\u8d44\u6e90\u9002\u914d\u5de5\u4f5c\u53f0",
  trainingTitle: "\u8bbe\u8ba1\u65b0\u4efb\u52a1\uff0c\u53ea\u8bad\u7ec3\u8bbe\u5907\u627f\u53d7\u5f97\u8d77\u7684\u90e8\u5206",
  trainingIntro: "\u8d44\u6e90\u95e8\u63a7\u8fde\u63a5\u793a\u8303\u3001\u8f7b\u91cf\u9002\u914d\u3001\u591a seed \u9a8c\u8bc1\u4e0e Pareto \u7b56\u7565\u664b\u7ea7\u3002",
  adaptValidate: "\u9a8c\u8bc1", adaptValidateSmall: "\u4efb\u52a1 + \u9884\u7b97", adaptCollect: "\u91c7\u96c6", adaptCollectSmall: "MuJoCo \u793a\u8303", adaptTrain: "\u9002\u914d", adaptTrainSmall: "\u5c0f\u578b\u6a21\u5757", adaptEvaluate: "\u9a8c\u8bc1", adaptEvaluateSmall: "\u72ec\u7acb seed", adaptPackage: "\u5f52\u6863", adaptPackageSmall: "\u6a21\u578b + \u8bc1\u636e",
  taskPreview: "\u4efb\u52a1\u9884\u89c8", lossCurve: "\u5f52\u4e00\u5316\u635f\u5931", stage: "\u9636\u6bb5", itemProgress: "\u5b50\u8fdb\u5ea6", updatedParams: "\u66f4\u65b0\u53c2\u6570", estimatedRam: "\u9884\u4f30\u5cf0\u503c\u5185\u5b58", beforeRun: "\u8fd0\u884c\u524d\u4f30\u7b97", liveRam: "\u5b9e\u65f6\u8fdb\u7a0b\u5185\u5b58", peakRam: "\u5b9e\u6d4b\u5cf0\u503c\u5185\u5b58", measured: "\u5b9e\u6d4b", demoYield: "\u793a\u8303\u6210\u529f", strictSuccess: "\u4e25\u683c\u6210\u529f", processTrace: "\u8fc7\u7a0b\u8f68\u8ff9", noAdaptEvents: "\u6682\u65e0\u9002\u914d\u4e8b\u4ef6\u3002", promotionBoard: "\u7b56\u7565\u664b\u7ea7",
  taskForge: "\u4efb\u52a1\u5de5\u574a", sourceObject: "\u6e90\u7269\u4f53", targetRegion: "\u76ee\u6807\u533a\u57df", englishInstruction: "\u82f1\u6587\u6307\u4ee4", registerTask: "\u6ce8\u518c\u4efb\u52a1", registeredTask: "\u5df2\u6ce8\u518c\u4efb\u52a1", resourceOrchestrator: "\u8d44\u6e90\u7f16\u6392\u5668", deviceProfile: "\u8bbe\u5907\u6863\u4f4d", viewerDuringCollection: "\u91c7\u96c6\u65f6\u6253\u5f00 MuJoCo viewer", estimateResources: "\u4f30\u7b97\u5e76\u9a8c\u8bc1\u8d44\u6e90", resourceGate: "\u8d44\u6e90\u95e8\u63a7", startAdaptation: "\u91c7\u96c6 + \u9002\u914d", showLog: "\u663e\u793a\u8fc7\u7a0b\u65e5\u5fd7", adaptationStarted: "\u4f4e\u8d44\u6e90\u9002\u914d\u4f5c\u4e1a\u5df2\u542f\u52a8\u3002", taskRegistered: "\u65b0 MuJoCo \u4efb\u52a1\u5df2\u6ce8\u518c\u3002", estimateReady: "\u8d44\u6e90\u4f30\u7b97\u5df2\u66f4\u65b0\u3002"
});

Object.assign(translations.en, {
  navTraining: "Low-resource adapt",
  trainingRuns: "Train / adapt", adaptationRuns: "Low-resource adaptation",
  trainingEyebrow: "Low-resource adaptation studio",
  trainingTitle: "Design a task. Fit only what the device can afford.",
  trainingIntro: "Resource gating connects demonstrations, lightweight adaptation, multi-seed verification and Pareto policy promotion.",
  adaptValidate: "VALIDATE", adaptValidateSmall: "task + budget", adaptCollect: "COLLECT", adaptCollectSmall: "MuJoCo demos", adaptTrain: "ADAPT", adaptTrainSmall: "small module", adaptEvaluate: "VERIFY", adaptEvaluateSmall: "held-out seed", adaptPackage: "PACKAGE", adaptPackageSmall: "model + evidence",
  taskPreview: "Task preview", lossCurve: "Normalised loss", stage: "Stage", itemProgress: "Item", updatedParams: "Updated parameters", estimatedRam: "Estimated peak RAM", beforeRun: "before run", liveRam: "Live process RAM", peakRam: "Measured peak RAM", measured: "measured", demoYield: "Demo yield", strictSuccess: "strict success", processTrace: "Process trace", noAdaptEvents: "No adaptation events yet.", promotionBoard: "Policy promotion",
  taskForge: "Task forge", sourceObject: "Source object", targetRegion: "Target region", englishInstruction: "English instruction", registerTask: "Register task", registeredTask: "Registered task", resourceOrchestrator: "Resource orchestrator", deviceProfile: "Device profile", viewerDuringCollection: "Open MuJoCo viewer during collection", estimateResources: "Estimate and validate resources", resourceGate: "Resource gate", startAdaptation: "Collect + adapt", showLog: "Show process log", adaptationStarted: "Low-resource adaptation job started.", taskRegistered: "New MuJoCo task registered.", estimateReady: "Resource estimate updated."
});

Object.assign(translations.zh, {
  pairedOptimizer: "配对优化器",
  pairedOptimizerIntro: "所有候选共享一份示范数据和同一组留出 seed。",
  reuseVerifiedDemos: "指纹一致时复用已验证示范（命中后跳过采集 viewer）",
  runFairComparison: "运行公平对比",
  arenaStarted: "配对候选实验已启动。",
});

Object.assign(translations.en, {
  pairedOptimizer: "Paired optimizer",
  pairedOptimizerIntro: "Every candidate shares one demonstration set and one held-out seed sequence.",
  reuseVerifiedDemos: "Reuse verified demos when the fingerprint matches",
  runFairComparison: "Run fair comparison",
  arenaStarted: "Paired candidate experiment started.",
});

const taskLabels = {
  place_blue_cube_blue_pad: { zh: "蓝色方块 → 蓝色目标区", en: "Blue cube → blue target" },
  place_blue_cube_red_pad: { zh: "蓝色方块 → 红色目标区", en: "Blue cube → red target" },
  place_red_cube_red_pad: { zh: "红色方块 → 红色目标区", en: "Red cube → red target" },
  move_leftmost_cube_to_bowl: { zh: "最左方块 → 白色碗", en: "Leftmost cube → white bowl" }
};

const datasetLabels = {
  blue_blue_100: { zh: "蓝块→蓝区 · 100 条示范", en: "Blue→blue · 100 demos" },
  blue_red_100: { zh: "蓝块→红区 · 100 条示范", en: "Blue→red · 100 demos" },
  red_red_80: { zh: "红块→红区 · 80 条示范", en: "Red→red · 80 demos" },
  leftmost_50: { zh: "最左方块→白碗 · 50 条示范", en: "Leftmost→bowl · 50 demos" }
};

const routeParams = new URLSearchParams(location.search);
const validViews = new Set(["live", "benchmark", "governance", "portfolio", "releases", "training", "experiments", "evidence", "legacy"]);
const validLifecycles = new Set(["retained", "lightweight_probe", "learned_baseline", "rejected", "reference", "control_or_data"]);
const initialView = validViews.has(routeParams.get("view")) ? routeParams.get("view") : "live";
const initialLifecycle = validLifecycles.has(routeParams.get("lifecycle")) ? routeParams.get("lifecycle") : "retained";
const state = { language: localStorage.getItem("research-platform-language") || "zh", config: null, simulation: null, training: null, adaptation: null, adaptationEstimate: null, arenaEstimate: null, adaptationControlsSynced: false, benchmark: null, analytics: null, runs: [], health: null, studies: [], governanceSummary: null, selectedStudy: null, portfolio: null, selectedClaimId: routeParams.get("claim"), selectedLifecycle: initialLifecycle, releaseData: null, selectedReleaseId: routeParams.get("release"), selectedRun: null, activeView: initialView, frameSequence: -1, frameBusy: false };

const $ = id => document.getElementById(id);
const t = key => translations[state.language][key] || key;
const formatNumber = (value, digits = 3) => Number.isFinite(value) ? Number(value).toFixed(digits) : "--";
const escapeHtml = value => String(value ?? "").replace(/[&<>'"]/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
const containsHan = value => /\p{Script=Han}/u.test(String(value ?? ""));
const englishSafe = (value, fallback) => state.language === "en" && containsHan(value) ? fallback : value;
let appliedLanguage = null;

function syncLegacyLanguage() {
  const frame = $("legacyFrame");
  if (!frame || frame.src.endsWith("about:blank")) return;
  try {
    const document = frame.contentDocument;
    const toggle = document?.getElementById("languageToggle");
    if (!document || !toggle) return;
    const desired = state.language === "zh" ? "zh-CN" : "en";
    if (document.documentElement.lang !== desired) toggle.click();
    const updateLabel = () => { toggle.textContent = document.documentElement.lang === "en" ? "EN" : "中文"; };
    updateLabel();
    if (!toggle.dataset.currentLanguageLabel) {
      toggle.dataset.currentLanguageLabel = "true";
      toggle.addEventListener("click", () => setTimeout(updateLabel, 0));
    }
  } catch {}
}

function setLanguage(language) {
  if (appliedLanguage) {
    $("studyTitleInput").dataset[`${appliedLanguage}Draft`] = $("studyTitleInput").value;
    $("studyHypothesisInput").dataset[`${appliedLanguage}Draft`] = $("studyHypothesisInput").value;
  }
  state.language = language;
  localStorage.setItem("research-platform-language", language);
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  $("studyTitleInput").value = $("studyTitleInput").dataset[`${language}Draft`] || t("defaultStudyTitle");
  $("studyHypothesisInput").value = $("studyHypothesisInput").dataset[`${language}Draft`] || t("defaultStudyHypothesis");
  document.querySelectorAll("[data-command-en]").forEach(button => { button.dataset.command = button.dataset[`command${language === "zh" ? "Zh" : "En"}`]; });
  if (containsHan($("commandInput").value) && language === "en") $("commandInput").value = "";
  $("languageSwitch").textContent = language === "en" ? "EN" : "中文";
  $("languageSwitch").setAttribute("aria-label", t("switchLanguage"));
  $("languageSwitch").title = t("switchLanguage");
  $("toast").textContent = "";
  $("toast").className = "toast";
  appliedLanguage = language;
  renderTaskOptions();
  renderDatasetOptions();
  renderStatus();
  renderBenchmark();
  renderGovernance();
  renderPortfolio();
  renderReleases();
  renderAdaptation();
  renderAnalytics();
  renderRuns();
  renderHealth();
  syncLegacyLanguage();
}

function showToast(message, error = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast visible${error ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = "toast"; }, 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function switchView(view) {
  state.activeView = view;
  document.querySelectorAll(".nav-item").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  document.querySelectorAll(".view").forEach(section => section.classList.toggle("active", section.id === `view-${view}`));
  if (view === "legacy" && $("legacyFrame").src.endsWith("about:blank")) {
    $("legacyFrame").src = state.config?.legacy_path || "/docs/integrated_research_showcase.html";
  }
  if (view === "training") { refreshAdaptation(); requestAnimationFrame(drawLossChart); }
  if (view === "governance") refreshStudies();
  if (view === "portfolio") refreshPortfolio();
  if (view === "releases") refreshReleases();
  if (view === "experiments") refreshResearchRecords();
  syncRoute();
}

function syncRoute() {
  const url = new URL(location.href);
  state.activeView === "live" ? url.searchParams.delete("view") : url.searchParams.set("view", state.activeView);
  if (state.activeView === "portfolio" && state.selectedClaimId) {
    url.searchParams.set("claim", state.selectedClaimId);
    url.searchParams.set("lifecycle", state.selectedLifecycle);
  } else {
    url.searchParams.delete("claim");
    url.searchParams.delete("lifecycle");
  }
  if (state.activeView === "releases" && state.selectedReleaseId) url.searchParams.set("release", state.selectedReleaseId);
  else url.searchParams.delete("release");
  url.searchParams.delete("scene");
  history.replaceState(null, "", url);
}

function renderTaskOptions() {
  if (!state.config) return;
  const current = $("taskSelect").value;
  $("taskSelect").innerHTML = Object.keys(state.config.tasks).map(id => `<option value="${id}">${taskLabels[id]?.[state.language] || id}</option>`).join("");
  if (current && state.config.tasks[current]) $("taskSelect").value = current;
}

function renderDatasetOptions() {
  if (!state.config || !$('datasetSelect')) return;
  const current = $("datasetSelect").value;
  $("datasetSelect").innerHTML = Object.entries(state.config.datasets).filter(([, item]) => item.available).map(([id]) => `<option value="${id}">${datasetLabels[id]?.[state.language] || id}</option>`).join("");
  if (current && state.config.datasets[current]) $("datasetSelect").value = current;
}

function statusText(status) {
  return t(status || "idle");
}

function policyText(policy) {
  return policy === "rgb_grounded" ? t("policyRgb") : policy === "structured_state" ? t("policyState") : policy;
}

function renderBenchmark() {
  const benchmark = state.benchmark;
  if (!benchmark) return;
  $("benchmarkState").textContent = statusText(benchmark.status);
  $("benchmarkState").className = `training-state ${benchmark.status}`;
  $("benchmarkProgress").style.width = `${Math.max(0, Math.min(100, (benchmark.progress || 0) * 100))}%`;
  $("benchmarkEpisodes").textContent = `${benchmark.completed_episodes || 0} / ${benchmark.total_episodes || 0}`;
  const policyMetrics = benchmark.policy_metrics || [];
  const rgbMetric = policyMetrics.find(row => row.policy === "rgb_grounded");
  const stateMetric = policyMetrics.find(row => row.policy === "structured_state");
  $("benchmarkSuccessRate").textContent = policyMetrics.length ? `${rgbMetric ? `${formatNumber(rgbMetric.success_rate * 100, 0)}%` : "--"} / ${stateMetric ? `${formatNumber(stateMetric.success_rate * 100, 0)}%` : "--"}` : "--";
  const paired = benchmark.paired_summary || {};
  const disagreements = (paired.rgb_only || 0) + (paired.state_only || 0);
  $("benchmarkMeanError").textContent = paired.pairs ? `${disagreements} / ${paired.pairs}` : "--";
  $("benchmarkElapsed").textContent = `${formatNumber(benchmark.elapsed, 1)} s`;
  $("benchmarkId").textContent = benchmark.benchmark_id || "--";
  $("benchmarkCurrent").textContent = benchmark.current_task ? `${taskLabels[benchmark.current_task]?.[state.language] || benchmark.current_task} · seed ${benchmark.current_seed} · ${policyText(benchmark.current_policy)}` : "--";
  const results = benchmark.results || [];
  $("benchmarkRows").innerHTML = results.length ? results.map(row => `
    <tr>
      <td>${escapeHtml(taskLabels[row.task]?.[state.language] || row.task)}</td>
      <td>${escapeHtml(row.seed)}</td>
      <td>${escapeHtml(row.policy === "rgb_grounded" ? "RGB" : "STATE")}</td>
      <td class="${row.success ? "result-success" : "result-failure"}">${row.success ? t("success") : t("failure")}</td>
      <td>${Number.isFinite(row.target_distance) ? `${formatNumber(row.target_distance * 1000, 1)} mm` : "--"}</td>
      <td>${formatNumber(row.elapsed, 1)} s</td>
    </tr>`).join("") : `<tr><td colspan="6">${t("noBenchmarkResults")}</td></tr>`;

  const metricCards = policyMetrics.map(metric => `
    <div class="paired-policy-card ${metric.policy}">
      <div><b>${metric.policy === "rgb_grounded" ? "RGB" : "STATE"}</b><strong>${formatNumber(metric.success_rate * 100, 1)}%</strong></div>
      <span>${t("ci95")} ${formatNumber(metric.ci95_low * 100, 1)}–${formatNumber(metric.ci95_high * 100, 1)}%</span>
      <div class="rate-track"><i style="width:${Math.max(0, Math.min(100, metric.success_rate * 100))}%"></i></div>
      <small>${metric.successes}/${metric.episodes} · ${Number.isFinite(metric.mean_target_error) ? `${formatNumber(metric.mean_target_error * 1000, 1)} mm` : "--"}${Number.isFinite(metric.mean_grounding_error) ? ` · ${t("groundingError")} ${formatNumber(metric.mean_grounding_error * 1000, 1)} mm` : ""}</small>
    </div>`).join("");
  const pairMatrix = paired.pairs ? `
    <div class="pair-matrix">
      <span><b>${paired.both_success}</b>${t("bothSuccess")}</span>
      <span><b>${paired.rgb_only}</b>${t("rgbOnly")}</span>
      <span><b>${paired.state_only}</b>${t("stateOnly")}</span>
      <span><b>${paired.both_fail}</b>${t("bothFail")}</span>
    </div>` : `<div class="pair-empty">${t("noPairedEvidence")}</div>`;
  $("pairedComparison").innerHTML = metricCards || pairMatrix ? `${metricCards}${pairMatrix}` : `<div class="pair-empty">${t("noPairedEvidence")}</div>`;
}

const gateLabels = {
  protocol: "gateProtocol",
  execution: "gateExecution",
  pairing: "gatePairing",
  artifacts: "gateArtifacts",
  success: "gateSuccess",
  target_error: "gateTarget",
  grounding: "gateGrounding",
  uncertainty: "gateUncertainty",
};

function verdictText(verdict) {
  const labels = {
    locked: "verdictLocked",
    executing: "verdictExecuting",
    ready_to_report: "verdictReady",
    needs_more_evidence: "verdictNeedsEvidence",
    execution_failed: "verdictFailed",
  };
  return t(labels[verdict] || verdict);
}

function updateStudyWorkload() {
  const tasks = document.querySelectorAll(".governance-task-checks input:checked").length;
  const seeds = Number($("studyCountInput").value) || 0;
  $("studyWorkload").textContent = `${tasks * seeds * 2} episodes · ${tasks * seeds} pairs`;
}

function gateValue(gate, value) {
  if (value === null || value === undefined) return "--";
  if (Array.isArray(value)) return value.length ? value.map(item => gateValue(gate, item)).join(" / ") : "--";
  if (gate.id === "success" || gate.id === "uncertainty") return `${formatNumber(Number(value) * 100, 1)}%`;
  if (gate.id === "target_error" || gate.id === "grounding") return `${formatNumber(Number(value), 1)} mm`;
  return String(value);
}

function renderGovernance() {
  $("studyCount").textContent = state.governanceSummary?.total || 0;
  $("traceFramework").textContent = state.governanceSummary?.framework || state.config?.governance_framework || "TRACE-1.0";
  const studies = state.studies || [];
  $("studyList").innerHTML = studies.length ? studies.map(study => {
    const evaluation = study.evaluation;
    const passed = evaluation.gates.filter(gate => gate.status === "pass").length;
    return `<button class="study-row ${state.selectedStudy?.study_id === study.study_id ? "selected" : ""}" type="button" data-study-id="${escapeHtml(study.study_id)}">
      <span class="study-verdict ${evaluation.verdict}">${escapeHtml(verdictText(evaluation.verdict))}</span>
      <strong>${escapeHtml(englishSafe(study.protocol.title, `${t("localLanguageStudy")} · ${study.study_id}`))}</strong>
      <small>${escapeHtml(study.protocol_hash.slice(0, 12))} · ${study.protocol.expected_episodes} episodes · ${passed}/8 gates</small>
    </button>`;
  }).join("") : `<div class="empty-study">${t("noStudies")}</div>`;

  const study = state.selectedStudy;
  $("emptyDossier").hidden = Boolean(study);
  $("studyDossier").hidden = !study;
  document.querySelectorAll("[data-trace-stage]").forEach(node => { node.className = ""; });
  if (!study) {
    document.querySelector('[data-trace-stage="target"]')?.classList.add("active");
    return;
  }

  const evaluation = study.evaluation;
  Object.entries(evaluation.stages).forEach(([name, status]) => document.querySelector(`[data-trace-stage="${name}"]`)?.classList.add(status));
  $("studyVerdict").textContent = verdictText(evaluation.verdict);
  $("studyVerdict").className = evaluation.verdict;
  $("studyHash").textContent = `SHA256 ${study.protocol_hash.slice(0, 12)}`;
  $("dossierTitle").textContent = englishSafe(study.protocol.title, `${t("localLanguageStudy")} · ${study.study_id}`);
  $("dossierHypothesis").textContent = englishSafe(study.protocol.hypothesis, t("localLanguageHypothesis"));
  $("downloadStudyMemo").href = `/api/studies/${encodeURIComponent(study.study_id)}/memo.md`;
  const isBusy = ["starting", "running"].includes(state.benchmark?.status);
  $("launchStudy").disabled = isBusy;
  $("launchStudy").textContent = study.launches.length ? t("rerunProtocol") : t("launchProtocol");
  const passed = evaluation.gates.filter(gate => gate.status === "pass").length;
  $("gateScore").textContent = `${passed} / ${evaluation.gates.length}`;
  $("dossierMetrics").innerHTML = `
    <div><span>${t("expectedEpisodes")}</span><strong>${study.protocol.expected_episodes}</strong></div>
    <div><span>${t("latestBenchmark")}</span><strong>${escapeHtml(evaluation.latest_benchmark_id || "--")}</strong></div>
    <div><span>${t("completedPairs")}</span><strong>${evaluation.paired_summary?.pairs || 0}</strong></div>
    <div><span>${t("passedGates")}</span><strong>${passed}/${evaluation.gates.length}</strong></div>`;
  $("gateList").innerHTML = evaluation.gates.map(gate => `
    <div class="gate-row ${gate.status}">
      <span class="gate-symbol">${gate.status === "pass" ? "✓" : gate.status === "fail" ? "×" : "·"}</span>
      <div><strong>${escapeHtml(t(gateLabels[gate.id]))}</strong><small>${escapeHtml(gateValue(gate, gate.observed))} / ${escapeHtml(gateValue(gate, gate.threshold))}</small></div>
      <b>${escapeHtml(t(gate.status === "pass" ? "gatePass" : gate.status === "fail" ? "gateFail" : "gatePending"))}</b>
    </div>`).join("");
}

async function refreshStudies() {
  try {
    const payload = await api("/api/studies");
    const selectedId = state.selectedStudy?.study_id;
    state.studies = payload.studies || [];
    state.governanceSummary = payload.summary;
    state.selectedStudy = state.studies.find(study => study.study_id === selectedId) || state.studies[0] || null;
    renderGovernance();
  } catch (error) {
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

function studyRequest() {
  return {
    title: $("studyTitleInput").value,
    hypothesis: $("studyHypothesisInput").value,
    tasks: [...document.querySelectorAll(".governance-task-checks input:checked")].map(input => input.value),
    seed_start: Number($("studySeedInput").value),
    seeds_per_task: Number($("studyCountInput").value),
    speed: 3.0,
    criteria: {
      min_success_rate: Number($("studySuccessInput").value) / 100,
      max_target_error_mm: Number($("studyTargetInput").value),
      max_grounding_error_mm: Number($("studyGroundingInput").value),
      max_ci_width: Number($("studyCiInput").value) / 100,
    },
  };
}

async function lockProtocol() {
  try {
    const study = await api("/api/studies", {method: "POST", body: JSON.stringify(studyRequest())});
    state.selectedStudy = study;
    await refreshStudies();
    showToast(t("protocolCreated"));
  } catch (error) {
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

async function launchStudy() {
  if (!state.selectedStudy) return;
  try {
    const payload = await api(`/api/studies/${encodeURIComponent(state.selectedStudy.study_id)}/launch`, {method: "POST", body: "{}"});
    state.benchmark = payload.benchmark;
    state.selectedStudy = payload.study;
    renderBenchmark();
    renderGovernance();
    showToast(t("protocolLaunched"));
  } catch (error) {
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

const claimStatusLabels = {
  reportable: "claimReportable",
  bounded: "claimBounded",
  negative: "claimNegative",
  blocked: "claimBlocked",
};

const lifecycleLabels = {
  retained: "lifecycleRetained",
  lightweight_probe: "lifecycleProbe",
  learned_baseline: "lifecycleBaseline",
  rejected: "lifecycleRejected",
  reference: "lifecycleReference",
  control_or_data: "lifecycleControl",
};

function renderPortfolio() {
  const portfolio = state.portfolio;
  if (!portfolio) return;
  const claims = portfolio.claims || [];
  const selectedClaim = claims.find(claim => claim.id === state.selectedClaimId) || claims[0];
  state.selectedClaimId = selectedClaim?.id || null;
  const summary = portfolio.summary;
  $("portfolioClaimCount").textContent = summary.claims;
  $("portfolioMethodCount").textContent = summary.methods;
  $("portfolioIntegrityScore").textContent = `${summary.integrity_passed}/${summary.integrity_total}`;
  $("portfolioFramework").textContent = portfolio.framework;
  $("portfolioDigest").textContent = `SHA256 ${portfolio.source_digest.slice(0, 8)}`;
  $("verifiedVideoCount").textContent = `${summary.verified_videos} VIDEO`;

  $("claimList").innerHTML = claims.map(claim => `
    <button class="claim-row ${claim.id === state.selectedClaimId ? "selected" : ""}" type="button" data-claim-id="${escapeHtml(claim.id)}">
      <span class="claim-state ${claim.status}">${escapeHtml(t(claimStatusLabels[claim.status] || claim.status))}</span>
      <strong>${escapeHtml(claim.title[state.language])}</strong>
      <span class="claim-readiness-bar"><i style="width:${claim.readiness}%"></i></span>
      <small>${claim.readiness}% · ${claim.evidence_ids.length} sources</small>
    </button>`).join("");

  if (selectedClaim) {
    $("claimStatus").textContent = t(claimStatusLabels[selectedClaim.status] || selectedClaim.status);
    $("claimStatus").className = selectedClaim.status;
    $("claimReadiness").textContent = `${selectedClaim.readiness}%`;
    $("claimTitle").textContent = selectedClaim.title[state.language];
    $("claimAllowed").textContent = selectedClaim.allowed[state.language];
    $("claimBlocked").textContent = selectedClaim.blocked[state.language];
    $("claimNext").textContent = selectedClaim.next_action[state.language];
    $("claimMetrics").innerHTML = selectedClaim.metrics.map(metric => `<div><span>${escapeHtml(metric.label)}</span><strong>${escapeHtml(metric.value)}</strong></div>`).join("");
    const passed = selectedClaim.gates.filter(gate => gate.status === "pass").length;
    $("claimGateScore").textContent = `${passed}/${selectedClaim.gates.length}`;
    $("claimGates").innerHTML = selectedClaim.gates.map(gate => `
      <div class="claim-gate ${gate.status}"><span>${gate.status === "pass" ? "✓" : gate.status === "fail" ? "×" : "·"}</span><strong>${escapeHtml(gate.label)}</strong><b>${escapeHtml(t(gate.status === "pass" ? "gatePass" : gate.status === "fail" ? "gateFail" : "gatePending"))}</b></div>`).join("");
    $("claimSources").innerHTML = selectedClaim.evidence_ids.map(sourceId => {
      const source = portfolio.sources.find(item => item.id === sourceId);
      if (!source) return "";
      return `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener"><span>${escapeHtml(source.id)}</span><strong>${escapeHtml(source.path)}</strong><small>${escapeHtml(source.sha256.slice(0, 12))}</small></a>`;
    }).join("");
    const traceButton = $("openClaimStudy");
    traceButton.hidden = !selectedClaim.study_id;
    traceButton.dataset.studyId = selectedClaim.study_id || "";
  }

  const lifecycleOrder = ["retained", "lightweight_probe", "learned_baseline", "rejected", "reference", "control_or_data"];
  $("lifecycleFilters").innerHTML = lifecycleOrder.map(id => `<button class="${state.selectedLifecycle === id ? "active" : ""}" type="button" data-lifecycle="${id}"><span>${summary.lifecycles[id] || 0}</span>${escapeHtml(t(lifecycleLabels[id]))}</button>`).join("");
  const visibleMethods = portfolio.methods.filter(method => method.lifecycle === state.selectedLifecycle);
  $("methodList").innerHTML = visibleMethods.length ? visibleMethods.map(method => {
    const heldout = method.heldout_success ? `${method.heldout_success.successes}/${method.heldout_success.episodes}` : "--";
    const params = method.trainable_params === null ? "--" : method.trainable_params.toLocaleString();
    return `<article class="method-row ${method.outcome}"><div><strong>${escapeHtml(method.name)}</strong><small>${escapeHtml(method.stage)}</small></div><span><b>${heldout}</b><small>held-out</small></span><span><b>${params}</b><small>trainable</small></span></article>`;
  }).join("") : `<div class="empty-study">${t("noMethods")}</div>`;
  $("integrityList").innerHTML = portfolio.integrity.map(gate => `<div class="integrity-row ${gate.status}"><span>${gate.status === "pass" ? "✓" : "×"}</span><strong>${escapeHtml(gate.label)}</strong><b>${escapeHtml(gate.observed || t(gate.status === "pass" ? "gatePass" : "gateFail"))}</b></div>`).join("");
}

async function refreshPortfolio() {
  try {
    state.portfolio = await api("/api/portfolio");
    renderPortfolio();
  } catch (error) {
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

const releaseGateLabels = {
  portfolio_integrity: "releaseGatePortfolio",
  reportable_claim: "releaseGateClaim",
  trace_capture: "releaseGateTrace",
  ledger_capture: "releaseGateLedger",
  quiescent_state: "releaseGateIdle",
};

function releaseStatusText(status) {
  const labels = {
    verified_current: "releaseVerifiedCurrent",
    verified_snapshot: "releaseVerifiedSnapshot",
    corrupted: "releaseCorrupted",
  };
  return t(labels[status] || status);
}

function renderReleases() {
  const data = state.releaseData;
  if (!data) return;
  const preview = data.preview;
  const releases = data.releases || [];
  const selected = releases.find(release => release.release_id === state.selectedReleaseId) || releases[0] || null;
  state.selectedReleaseId = selected?.release_id || null;
  $("releaseReadiness").textContent = t(preview.ready ? "releaseReady" : "releaseBlocked");
  $("releaseReadiness").className = preview.ready ? "ready" : "blocked";
  $("releaseCount").textContent = data.summary.total;
  $("releaseFileCount").textContent = preview.bundle_file_count;
  $("releaseFramework").textContent = preview.framework;
  $("releasePortfolioDigest").textContent = preview.portfolio_digest;
  $("createRelease").disabled = !preview.ready;
  $("releaseRegistryStatus").textContent = `${data.summary.total} RECORDS`;
  $("releaseGateRail").innerHTML = preview.gates.map((gate, index) => `
    <div class="${gate.status}"><b>${String(index + 1).padStart(2, "0")}</b><span>${escapeHtml(t(releaseGateLabels[gate.id] || gate.label))}</span><strong>${gate.status === "pass" ? "✓" : "×"}</strong></div>`).join("");
  $("releaseList").innerHTML = releases.length ? releases.map(release => `
    <button class="release-row ${release.release_id === state.selectedReleaseId ? "selected" : ""}" type="button" data-release-id="${escapeHtml(release.release_id)}">
      <span class="release-state ${release.status}">${escapeHtml(releaseStatusText(release.status))}</span>
      <strong>${escapeHtml(englishSafe(release.label, `${t("localLanguageRelease")} · ${release.release_id}`))}</strong>
      <small>${escapeHtml(release.created_at)} · ${escapeHtml(release.manifest_hash.slice(0, 10))}</small>
    </button>`).join("") : `<div class="empty-study">${t("noReleases")}</div>`;

  $("emptyReleaseDossier").hidden = Boolean(selected);
  $("releaseDossier").hidden = !selected;
  if (!selected) return;
  $("releaseStatus").textContent = releaseStatusText(selected.status);
  $("releaseStatus").className = selected.status;
  $("releaseManifestHash").textContent = `SHA256 ${selected.manifest_hash.slice(0, 16)}`;
  $("releaseDossierTitle").textContent = englishSafe(selected.label, `${t("localLanguageRelease")} · ${selected.release_id}`);
  $("releaseDossierNote").textContent = englishSafe(selected.note, t("localLanguageRelease")) || "--";
  $("downloadReleaseManifest").href = selected.manifest_url;
  $("downloadReleaseReadme").href = selected.readme_url;
  $("releaseMetrics").innerHTML = `
    <div><span>${t("bundleFiles")}</span><strong>${selected.files.length}</strong></div>
    <div><span>${t("frozenClaims")}</span><strong>${selected.claims.length}</strong></div>
    <div><span>TRACE</span><strong>${selected.trace_studies.length}</strong></div>
    <div><span>${t("allRuns")}</span><strong>${selected.ledger.analytics?.total_runs ?? "--"}</strong></div>`;
  $("releaseClaimCount").textContent = selected.claims.length;
  $("releaseClaims").innerHTML = selected.claims.map(claim => `
    <div class="release-claim"><span class="claim-state ${claim.status}">${escapeHtml(t(claimStatusLabels[claim.status] || claim.status))}</span><strong>${escapeHtml(claim.title[state.language])}</strong><b>${claim.readiness}%</b></div>`).join("");
  $("releaseVerificationScore").textContent = `${selected.verification.verified_files}/${selected.verification.total_files}`;
  $("releaseFiles").innerHTML = selected.files.map(file => `
    <div class="release-file ${selected.verification.files_valid ? "pass" : "fail"}"><span>${selected.verification.files_valid ? "✓" : "×"}</span><strong>${escapeHtml(file.path)}</strong><small>${escapeHtml(file.sha256.slice(0, 12))}</small></div>`).join("");
  const checks = [
    ["manifestIntegrity", selected.verification.manifest_valid],
    ["fileIntegrity", selected.verification.files_valid],
    ["portfolioMatch", selected.verification.current_portfolio],
    ["ledgerMatch", selected.verification.current_ledger],
  ];
  $("releaseDrift").innerHTML = checks.map(([label, passed]) => `<div class="${passed ? "pass" : "drift"}"><span>${t(label)}</span><strong>${passed ? "✓" : "△"}</strong></div>`).join("");
}

async function refreshReleases() {
  try {
    const data = await api("/api/releases");
    state.releaseData = data;
    state.selectedReleaseId = data.releases.some(release => release.release_id === state.selectedReleaseId) ? state.selectedReleaseId : data.releases[0]?.release_id || null;
    renderReleases();
    syncRoute();
  } catch (error) {
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

async function createEvidenceRelease() {
  try {
    const release = await api("/api/releases", {
      method: "POST",
      body: JSON.stringify({label: $("releaseLabelInput").value, note: $("releaseNoteInput").value}),
    });
    state.selectedReleaseId = release.release_id;
    await refreshReleases();
    showToast(t("releaseCreated"));
  } catch (error) {
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

function formatClock(seconds) {
  const value = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function renderAnalytics() {
  const analytics = state.analytics;
  if (!analytics) return;
  $("totalRuns").textContent = analytics.total_runs || 0;
  $("simulationRuns").textContent = analytics.simulations || 0;
  $("trainingRuns").textContent = (analytics.trainings || 0) + (analytics.adaptations || 0);
  $("benchmarkRuns").textContent = analytics.benchmarks || 0;
  const policies = analytics.policies || [];
  $("policyComparison").innerHTML = policies.length ? policies.map(row => `
    <div class="policy-row">
      <span>${row.policy === "rgb_grounded" ? t("policyRgb") : t("policyState")}</span>
      <strong>${formatNumber(row.success_rate * 100, 1)}%</strong>
      <small>n=${row.episodes} · ${Number.isFinite(row.mean_target_error) ? `${formatNumber(row.mean_target_error * 1000, 1)} mm` : "--"}</small>
    </div>`).join("") : `<div class="policy-empty">${state.language === "zh" ? "完成平台仿真后生成策略汇总。" : "Policy summaries appear after platform simulations complete."}</div>`;
  const trainings = analytics.latest_training || [];
  $("trainingComparison").innerHTML = trainings.length ? trainings.map(row => `<div class="training-summary"><strong>${escapeHtml(row.method)}</strong><span>train ${formatNumber(row.train_loss, 5)}</span><span>val ${formatNumber(row.val_loss, 5)}</span></div>`).join("") : `<span class="policy-empty">${state.language === "zh" ? "暂无平台训练记录。" : "No platform training records yet."}</span>`;
}

function runConfiguration(row) {
  const config = row.config || {};
  if (row.kind === "simulation") return `${taskLabels[config.task]?.[state.language] || config.task} · ${config.policy}`;
  if (row.kind === "training") return `${config.method} · ${config.dataset} · ${config.epochs} epochs`;
  if (row.kind === "adaptation") return `${config.task_id} · ${config.method} · ${config.profile}`;
  return `${(config.tasks || []).length} tasks · ${(config.policies || [config.policy]).length} policies · ${config.seeds_per_task || 0} seed(s)`;
}

function runMetric(row) {
  const metrics = row.metrics || {};
  if (row.kind === "simulation") return metrics.success === true ? `${t("success")} · ${formatNumber(metrics.target_distance * 1000, 1)} mm` : metrics.success === false ? t("failure") : "--";
  if (row.kind === "training") return Number.isFinite(metrics.val_loss) ? `val ${formatNumber(metrics.val_loss, 5)}` : "--";
  if (row.kind === "adaptation") return `${metrics.demonstration_successes || 0}/${metrics.episodes || 0} demos · ${Number(metrics.trainable_params || 0).toLocaleString()} params`;
  if (Array.isArray(metrics.policy_metrics) && metrics.policy_metrics.length) {
    return metrics.policy_metrics.map(metric => `${metric.policy === "rgb_grounded" ? "RGB" : "STATE"} ${formatNumber(metric.success_rate * 100, 1)}%`).join(" · ");
  }
  return Number.isFinite(metrics.success_rate) ? `${formatNumber(metrics.success_rate * 100, 1)}% · n=${metrics.episodes}` : "--";
}

function runKindText(kind) {
  const labels = {
    simulation: t("simulationRuns"),
    training: t("trainingRuns"),
    adaptation: t("adaptationRuns"),
    benchmark: t("benchmarkRuns"),
  };
  return labels[kind] || kind;
}

function renderRuns() {
  const rows = state.runs || [];
  $("runRows").innerHTML = rows.length ? rows.map(row => `
    <tr data-run-id="${escapeHtml(row.run_id)}" class="${state.selectedRun?.run_id === row.run_id ? "selected" : ""}">
      <td>${escapeHtml(row.run_id)}</td>
      <td>${escapeHtml(runKindText(row.kind))}</td>
      <td>${escapeHtml(runConfiguration(row))}</td>
      <td>${escapeHtml(statusText(row.status))}</td>
      <td>${escapeHtml(runMetric(row))}</td>
      <td>${escapeHtml(row.started_at ? new Date(row.started_at).toLocaleString(state.language === "zh" ? "zh-CN" : "en-GB", { hour12: false }) : "--")}</td>
    </tr>`).join("") : `<tr><td colspan="6">${t("noRuns")}</td></tr>`;
  if (state.selectedRun) renderSelectedRun();
}

function renderSelectedRun() {
  const row = state.selectedRun;
  if (!row) return;
  $("selectedRunKind").textContent = row.kind.toUpperCase();
  $("selectedRunId").textContent = row.run_id;
  const details = { status: statusText(row.status), ...row.config, ...row.metrics };
  $("selectedRunMetrics").innerHTML = Object.entries(details).filter(([, value]) => value !== null && typeof value !== "object").map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
  $("selectedArtifact").textContent = row.artifact || "--";
  const assets = row.assets || {};
  $("selectedAssets").innerHTML = Object.entries(assets).map(([name, url]) => `
    <figure><img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" loading="lazy"><figcaption>${name === "initial_top" ? t("visualInput") : t("finalState")}</figcaption></figure>`).join("");
  $("downloadJsonReport").href = `/api/runs/${encodeURIComponent(row.run_id)}/report.json`;
  $("downloadMarkdownReport").href = `/api/runs/${encodeURIComponent(row.run_id)}/report.md`;
  $("selectedCommand").textContent = row.command || "--";
}

function renderHealth() {
  const health = state.health;
  if (!health) return;
  $("healthStatus").textContent = health.status.toUpperCase();
  $("healthStatus").className = health.status;
  const readyDatasets = Object.values(health.datasets || {}).filter(Boolean).length;
  const allDatasets = Object.keys(health.datasets || {}).length;
  const episodeTotal = Object.values(health.dataset_episodes || {}).reduce((sum, value) => sum + value, 0);
  const rows = {
    "Platform": `v${health.platform_version}`,
    "Python / MuJoCo": `${health.python} / ${health.mujoco}`,
    "Assets": Object.values(health.assets || {}).every(Boolean) ? "READY" : "MISSING",
    "Datasets": `${readyDatasets} / ${allDatasets}`,
    "Demo episodes": episodeTotal,
    "TRACE studies": health.trace_studies || 0,
    "Disk free": `${formatNumber(health.disk_free_gb, 1)} GB`,
    "Ledger": health.ledger_path,
  };
  $("healthMetrics").innerHTML = Object.entries(rows).map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
}

async function refreshResearchRecords() {
  try {
    const params = new URLSearchParams();
    if ($("runKindFilter").value) params.set("kind", $("runKindFilter").value);
    if ($("runStatusFilter").value) params.set("status", $("runStatusFilter").value);
    params.set("limit", "200");
    const [records, health] = await Promise.all([api(`/api/runs?${params}`), api("/api/health")]);
    state.runs = records.runs;
    state.analytics = records.analytics;
    state.health = health;
    if (!state.selectedRun || !state.runs.some(row => row.run_id === state.selectedRun.run_id)) state.selectedRun = state.runs[0] || null;
    renderAnalytics();
    renderRuns();
    renderHealth();
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

function adaptationRequest() {
  return {
    task_id: $("adaptTaskSelect").value,
    method: $("adaptMethod").value,
    profile: $("adaptProfile").value,
    episodes: Number($("adaptEpisodes").value),
    epochs: Number($("adaptEpochs").value),
    evaluation_episodes: Number($("adaptEvaluationEpisodes").value),
    seed: Number($("adaptSeed").value),
    viewer: $("adaptViewer").checked,
  };
}

function arenaRequest() {
  return {
    ...adaptationRequest(),
    methods: [...document.querySelectorAll("#arenaMethods input:checked")].map(input => input.value),
    reuse_dataset: $("arenaReuseDataset").checked,
  };
}

function renderAdaptationTasks() {
  if (!state.adaptation) return;
  const select = $("adaptTaskSelect");
  const current = ["starting", "running"].includes(state.adaptation.status)
    ? state.adaptation.task_id
    : (select.value || state.adaptation.task_id);
  const tasks = state.adaptation.tasks || [];
  document.querySelector(".task-forge").open = tasks.length === 0;
  select.innerHTML = tasks.length
    ? tasks.map(task => `<option value="${escapeHtml(task.task_id)}">${escapeHtml(englishSafe(task.instruction, task.task_id) || task.task_id)}</option>`).join("")
    : `<option value="">${state.language === "zh" ? "\u8bf7\u5148\u6ce8\u518c\u65b0\u4efb\u52a1" : "Register a new task first"}</option>`;
  if (tasks.some(task => task.task_id === current)) select.value = current;
}

function renderPromotionBoard(portfolio = {}) {
  const candidates = portfolio.candidates || [];
  const champion = portfolio.champion_method;
  const promotion = portfolio.promotion || "no_evidence";
  const decisionLabels = state.language === "zh"
    ? { promoted: "通过晋级", needs_evidence: "证据不足", rejected: "未通过", no_evidence: "暂无证据" }
    : { promoted: "PROMOTED", needs_evidence: "NEEDS EVIDENCE", rejected: "REJECTED", no_evidence: "NO EVIDENCE" };
  $("promotionFramework").textContent = portfolio.framework || "RESOURCE-PARETO-1.0";
  $("promotionDecision").textContent = decisionLabels[promotion] || promotion.toUpperCase();
  $("promotionDecision").className = promotion === "promoted" ? "promoted" : promotion === "rejected" ? "rejected" : "";

  if (portfolio.improvement) {
    const delta = portfolio.improvement;
    $("promotionImprovement").textContent = state.language === "zh"
      ? `同 seed 对比 ${delta.reference_method}: +${formatNumber(delta.success_rate_points, 1)} pp / -${formatNumber(delta.target_error_reduction_mm, 1)} mm`
      : `PAIRED VS ${delta.reference_method}: +${formatNumber(delta.success_rate_points, 1)} PP / -${formatNumber(delta.target_error_reduction_mm, 1)} MM`;
  } else {
    $("promotionImprovement").textContent = state.language === "zh" ? "跨方法差值需使用相同 holdout seed" : "CROSS-METHOD DELTA REQUIRES MATCHED SEEDS";
  }

  $("promotionCandidates").innerHTML = candidates.length
    ? candidates.map(candidate => {
      const classes = ["promotion-candidate"];
      if ((portfolio.pareto_methods || []).includes(candidate.method)) classes.push("pareto");
      if (candidate.method === champion) classes.push("champion");
      const error = candidate.mean_target_error === null ? "--" : `${formatNumber(candidate.mean_target_error * 1000, 1)}mm`;
      return `<article class="${classes.join(" ")}" title="${escapeHtml(candidate.evaluation_seeds?.join(", ") || "no seeds")}"><strong>${escapeHtml(candidate.label)}</strong><span>${candidate.successes}/${candidate.evaluation_episodes}</span><span>${error}</span><span>${Number(candidate.trainable_params).toLocaleString()}p</span></article>`;
    }).join("")
    : `<span>${state.language === "zh" ? "完成适配后生成候选证据。" : "Complete an adaptation run to create candidate evidence."}</span>`;
}

function renderAdaptation() {
  const adaptation = state.adaptation;
  if (!adaptation) return;
  renderAdaptationTasks();
  const active = ["starting", "running"].includes(adaptation.status);
  if (!state.adaptationControlsSynced && adaptation.run_id) {
    $("adaptTaskSelect").value = adaptation.task_id || $("adaptTaskSelect").value;
    $("adaptMethod").value = adaptation.method || $("adaptMethod").value;
    $("adaptProfile").value = adaptation.requested_profile || $("adaptProfile").value;
    $("adaptEpisodes").value = adaptation.episodes || $("adaptEpisodes").value;
    $("adaptEpochs").value = adaptation.epochs || $("adaptEpochs").value;
    $("adaptEvaluationEpisodes").value = adaptation.evaluation_episodes || $("adaptEvaluationEpisodes").value;
    $("adaptSeed").value = adaptation.seed ?? $("adaptSeed").value;
    if (adaptation.mode === "arena" && adaptation.candidate_methods?.length) {
      document.querySelectorAll("#arenaMethods input").forEach(input => { input.checked = adaptation.candidate_methods.includes(input.value); });
      $("arenaReuseDataset").checked = Boolean(adaptation.dataset_cache_hit);
    }
    state.adaptationControlsSynced = true;
  }
  if (active && adaptation.mode === "arena" && adaptation.method) $("adaptMethod").value = adaptation.method;
  const storedEstimate = Object.keys(adaptation.estimated || {}).length ? adaptation.estimated : null;
  const arenaCandidateEstimate = adaptation.mode === "arena"
    ? storedEstimate?.candidate_estimates?.find(item => item.method === $("adaptMethod").value) || null
    : null;
  const estimate = state.adaptationEstimate || arenaCandidateEstimate || (adaptation.mode !== "arena" ? storedEstimate : null);
  const selectedTask = (adaptation.tasks || []).find(task => task.task_id === $("adaptTaskSelect").value);
  const seed = Number($("adaptSeed").value || 0);

  $("adaptationFramework").textContent = adaptation.framework;
  $("trainingState").textContent = statusText(adaptation.status).toUpperCase();
  $("trainingState").className = `training-state ${adaptation.status}`;
  $("hardwareClass").textContent = `${adaptation.hardware.cpu_logical} CPU / ${formatNumber(adaptation.hardware.ram_total_gb, 1)} GB / ${adaptation.hardware.recommended_profile.toUpperCase()}`;
  $("chartMethod").textContent = adaptation.methods?.[$("adaptMethod").value]?.label || adaptation.method;
  $("adaptationStage").textContent = (adaptation.stage || "ready").toUpperCase();
  $("trainEpoch").textContent = `${adaptation.current_item || 0} / ${adaptation.total_items || 0}`;
  $("trainLoss").textContent = formatNumber(adaptation.train_loss, 6);
  $("valLoss").textContent = formatNumber(adaptation.val_loss, 6);
  $("trainElapsed").textContent = `${formatNumber(adaptation.elapsed, 1)} s`;
  $("trainingProgress").style.width = `${Math.max(0, Math.min(100, (adaptation.progress || 0) * 100))}%`;
  $("modelPath").textContent = adaptation.model_path || "--";
  $("trainingLog").textContent = (adaptation.logs || []).join("\n");
  $("adaptLiveRam").textContent = `${formatNumber(adaptation.process_rss_mb, 1)} MB`;
  $("adaptPeakRam").textContent = `${formatNumber(adaptation.peak_rss_mb, 1)} MB`;
  $("adaptDemoYield").textContent = `${adaptation.collection_successes || 0} / ${adaptation.episodes || 0}`;

  const taskId = selectedTask?.task_id || adaptation.task_id || "";
  $("previewTaskId").textContent = taskId || "REGISTER A TASK";
  $("previewSeed").textContent = `SEED ${seed}`;
  $("adaptEvalResult").textContent = adaptation.evaluation_success_rate === null || adaptation.evaluation_success_rate === undefined
    ? "HOLDOUT --"
    : `HOLDOUT ${adaptation.evaluation_successes}/${adaptation.evaluation_episodes} / ${formatNumber((adaptation.evaluation_mean_target_error || 0) * 1000, 1)} MM`;
  $("adaptEvalResult").className = `preview-eval ${adaptation.evaluation_success === true ? "pass" : adaptation.evaluation_success === false ? "fail" : ""}`;
  if (taskId) {
    const previewUrl = `/api/adaptation/tasks/${encodeURIComponent(taskId)}/preview.png?seed=${seed}`;
    if ($("adaptationPreview").dataset.source !== previewUrl) {
      $("adaptationPreview").dataset.source = previewUrl;
      $("adaptationPreview").src = previewUrl;
    }
  }

  document.querySelectorAll("[data-adapt-stage]").forEach(node => node.className = "");
  const stages = ["validate", "collect", "train", "evaluate", "complete"];
  const stageIndex = stages.indexOf(adaptation.stage);
  document.querySelectorAll("[data-adapt-stage]").forEach((node, index) => {
    if (adaptation.status === "completed" || index < stageIndex) node.classList.add("done");
    if (index === stageIndex && adaptation.status !== "completed") node.classList.add("active");
    if (adaptation.status === "failed" && index === Math.max(0, stageIndex)) node.classList.add("failed");
  });

  if (estimate) {
    $("adaptParams").textContent = Number(estimate.trainable_params).toLocaleString();
    $("adaptParamSize").textContent = `${estimate.updated_parameter_mb} MB`;
    $("adaptEstimatedRam").textContent = `${estimate.estimated_peak_ram_mb} MB`;
    $("adaptEta").textContent = `ETA ${estimate.estimated_wall_seconds} s`;
    $("adaptationBoundary").textContent = estimate.truth_boundary;
    $("adaptGateStatus").textContent = estimate.gate.passed ? "PASS" : "BLOCKED";
    $("adaptResourceGate").className = `resource-gate ${estimate.gate.passed ? "pass" : "fail"}`;
    $("adaptGateDetail").textContent = estimate.gate.passed
      ? `${estimate.resolved_profile.toUpperCase()} / CPU ONLY / ${estimate.estimated_samples.toLocaleString()} SAMPLES / ${estimate.evaluation_episodes} HOLDOUT`
      : estimate.gate.reasons.join("; ");
  } else {
    $("adaptParams").textContent = "--";
    $("adaptParamSize").textContent = "-- MB";
    $("adaptEstimatedRam").textContent = "--";
    $("adaptEta").textContent = "ETA --";
    $("adaptGateStatus").textContent = "NOT CHECKED";
    $("adaptResourceGate").className = "resource-gate pending";
    $("adaptGateDetail").textContent = "--";
  }

  const events = adaptation.events || [];
  $("adaptationEvents").innerHTML = events.length
    ? events.slice(-12).map(event => `<article class="adapt-event ${escapeHtml(event.kind)}"><span>${escapeHtml(event.kind.toUpperCase())} / ${escapeHtml(event.time.slice(11, 19))}</span><strong>${escapeHtml(event.message)}</strong></article>`).join("")
    : `<span>${t("noAdaptEvents")}</span>`;
  $("adaptationEvents").scrollLeft = $("adaptationEvents").scrollWidth;
  $("startAdaptation").disabled = active || !estimate?.gate?.passed || !taskId;
  $("registerAdaptTask").disabled = active;
  $("estimateAdaptation").disabled = active || !taskId;
  $("stopAdaptation").disabled = !active;
  $("startArena").disabled = active || !taskId || arenaRequest().methods.length < 2;
  document.querySelectorAll("#arenaMethods input, #arenaReuseDataset").forEach(input => { input.disabled = active; });
  const latestArena = adaptation.latest_arena;
  const currentArena = adaptation.mode === "arena" && adaptation.paired_summary?.framework
    ? {paired: adaptation.paired_summary, candidates: adaptation.candidate_results || []}
    : latestArena?.metrics?.paired_summary?.framework
      ? {paired: latestArena.metrics.paired_summary, candidates: latestArena.metrics.candidate_results || []}
      : null;
  const arenaEstimate = state.arenaEstimate
    || (adaptation.mode === "arena" ? adaptation.estimated : null)
    || latestArena?.config?.resource_estimate
    || null;
  if (currentArena) {
    const paired = currentArena.paired;
    renderPromotionBoard({
      framework: paired.framework,
      champion_method: paired.champion_method,
      promotion: paired.promotion,
      pareto_methods: paired.pareto_methods,
      improvement: paired.paired_improvement,
      candidates: currentArena.candidates.filter(row => row.status === "completed"),
    });
  } else {
    renderPromotionBoard(adaptation.performance_portfolio || {});
  }
  if (arenaEstimate) {
    const cacheLabel = arenaEstimate.dataset_cache_hit ? (state.language === "zh" ? "缓存命中" : "CACHE HIT") : (state.language === "zh" ? "采集一次" : "ONE COLLECTION");
    $("arenaDetail").textContent = `${cacheLabel} / ${arenaEstimate.methods.length} candidates / ${arenaEstimate.evaluation_episodes} seeds / ${arenaEstimate.estimated_wall_seconds}s / ${arenaEstimate.dataset_fingerprint.slice(0, 12)}`;
  } else {
    $("arenaDetail").textContent = state.language === "zh" ? "至少选择两个候选；示范只采集一次。" : "Select at least two candidates; demonstrations are collected once.";
  }
  const efficiency = adaptation.arena_efficiency;
  $("arenaEfficiency").hidden = !efficiency;
  if (efficiency) {
    $("arenaEfficiency").textContent = state.language === "zh"
      ? `实测 ${formatNumber(efficiency.fresh_seconds, 2)}s -> ${formatNumber(efficiency.cached_seconds, 2)}s / 节省 ${formatNumber(efficiency.reduction_percent, 1)}% / 少执行 ${efficiency.cached_collection_runs_saved} 次采集`
      : `MEASURED ${formatNumber(efficiency.fresh_seconds, 2)}s -> ${formatNumber(efficiency.cached_seconds, 2)}s / ${formatNumber(efficiency.reduction_percent, 1)}% LOWER / ${efficiency.cached_collection_runs_saved} COLLECTIONS SAVED`;
    $("arenaEfficiency").title = efficiency.boundary;
  }
  drawLossChart();
}

async function refreshAdaptation() {
  try {
    state.adaptation = await api("/api/adaptation");
    renderAdaptation();
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

async function registerAdaptationTask() {
  try {
    const task = await api("/api/adaptation/tasks", {
      method: "POST",
      body: JSON.stringify({
        source: $("adaptSource").value,
        target: $("adaptTarget").value,
        instruction: $("adaptInstruction").value,
        complexity: $("adaptComplexity").value,
      }),
    });
    if (state.config) state.config.tasks[task.task_id] = task;
    await refreshAdaptation();
    $("adaptTaskSelect").value = task.task_id;
    state.adaptationEstimate = null;
    renderTaskOptions();
    renderAdaptation();
    showToast(t("taskRegistered"));
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

async function estimateAdaptation() {
  try {
    state.adaptationEstimate = await api("/api/adaptation/estimate", { method: "POST", body: JSON.stringify(adaptationRequest()) });
    renderAdaptation();
    showToast(t("estimateReady"));
    return state.adaptationEstimate;
  } catch (error) {
    state.adaptationEstimate = null;
    renderAdaptation();
    showToast(`${t("requestFailed")}: ${error.message}`, true);
    return null;
  }
}

async function startAdaptation() {
  try {
    state.arenaEstimate = null;
    if (!state.adaptationEstimate) await estimateAdaptation();
    if (!state.adaptationEstimate?.gate?.passed) return;
    state.adaptation = await api("/api/adaptation/start", { method: "POST", body: JSON.stringify(adaptationRequest()) });
    state.adaptationEstimate = state.adaptation.estimated;
    renderAdaptation();
    showToast(t("adaptationStarted"));
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

async function startArena() {
  try {
    const request = arenaRequest();
    state.adaptationEstimate = null;
    state.arenaEstimate = await api("/api/adaptation/arena/estimate", { method: "POST", body: JSON.stringify(request) });
    renderAdaptation();
    if (!state.arenaEstimate.gate.passed) {
      showToast(`${t("requestFailed")}: ${state.arenaEstimate.gate.reasons.join("; ")}`, true);
      return;
    }
    state.adaptation = await api("/api/adaptation/arena/start", { method: "POST", body: JSON.stringify(request) });
    state.arenaEstimate = state.adaptation.estimated;
    renderAdaptation();
    showToast(t("arenaStarted"));
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

function updateAdaptationInstruction() {
  const source = $("adaptSource").selectedOptions[0]?.textContent.toLowerCase() || "object";
  const target = $("adaptTarget").selectedOptions[0]?.textContent.toLowerCase() || "target";
  $("adaptInstruction").value = `place the ${source} on the ${target}`;
}

function renderStatus() {
  const sim = state.simulation;
  const training = state.training;
  if (sim) {
    $("simSeedLabel").textContent = `SEED ${sim.seed}`;
    $("simPolicyLabel").textContent = sim.policy === "rgb_grounded" ? "RGB GROUNDED" : "STATE REFERENCE";
    $("liveBadgeText").textContent = statusText(sim.status);
    $("liveBadgeText").parentElement.className = `live-badge ${sim.status}`;
    $("metricPhase").textContent = (sim.phase || "ready").toUpperCase();
    $("metricElapsed").textContent = `${formatNumber(sim.elapsed, 1)} s`;
    $("metricTarget").textContent = Number.isFinite(sim.target_distance) ? `${formatNumber(sim.target_distance * 1000, 1)} mm` : "--";
    $("metricHeight").textContent = Number.isFinite(sim.object_z) ? `${formatNumber(sim.object_z * 1000, 1)} mm` : "--";
    $("metricContacts").textContent = Math.round(sim.contact_count || 0);
    $("metricResult").textContent = sim.success === true ? t("success") : sim.success === false ? t("failure") : "--";
    $("simProgress").style.width = `${Math.max(0, Math.min(100, (sim.progress || 0) * 100))}%`;
    $("frameRate").textContent = `${formatNumber(sim.fps, 1)} FPS`;
    $("runtimeSource").textContent = sim.position_source || "--";
    $("runtimeObject").textContent = sim.source_name ? `${sim.source_name} · ${sim.source_position?.join(", ") || ""}` : "--";
    const stageMessage = $("stageMessage");
    if (["failed", "stopped"].includes(sim.status)) {
      stageMessage.classList.add("visible");
      stageMessage.querySelector("strong").textContent = sim.status === "failed" ? t("failed") : t("stopped");
      stageMessage.querySelector("span").textContent = sim.error || (state.language === "zh" ? "可以重置或输入新指令。" : "Reset or enter a new command.");
    } else if (sim.status === "idle") {
      stageMessage.classList.add("visible");
      stageMessage.querySelector("strong").textContent = t("readyTitle");
      stageMessage.querySelector("span").textContent = t("readyBody");
    } else {
      stageMessage.classList.remove("visible");
    }
  }
  if (training && state.adaptation) renderAdaptation();
  renderBenchmark();
  renderAnalytics();
}

function drawLossChart() {
  const canvas = $("lossChart");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(rect.width * ratio));
  const height = Math.max(220, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const pad = { left: 56 * ratio, right: 22 * ratio, top: 22 * ratio, bottom: 42 * ratio };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  if ($("adaptMethod").value === "registry_rgb_skill") {
    const stages = state.language === "zh"
      ? ["任务注册表", "RGB 标定定位", "结构化技能"]
      : ["TASK REGISTRY", "RGB LOCALISATION", "STRUCTURED SKILL"];
    ctx.fillStyle = "#0d8878";
    ctx.font = `700 ${11 * ratio}px Cascadia Mono, Consolas, monospace`;
    ctx.fillText("ZERO-GRADIENT COMPILE", pad.left, pad.top + 5 * ratio);
    const gap = 18 * ratio;
    const nodeWidth = (plotW - gap * 2) / 3;
    const nodeHeight = 58 * ratio;
    const nodeY = pad.top + plotH * 0.36;
    stages.forEach((label, index) => {
      const x = pad.left + index * (nodeWidth + gap);
      ctx.fillStyle = index === 2 ? "#e8f5f2" : "#f3f6f7";
      ctx.strokeStyle = index === 2 ? "#0d8878" : "#b9c6cc";
      ctx.lineWidth = 1.5 * ratio;
      ctx.fillRect(x, nodeY, nodeWidth, nodeHeight);
      ctx.strokeRect(x, nodeY, nodeWidth, nodeHeight);
      ctx.fillStyle = "#153242";
      ctx.font = `650 ${9 * ratio}px Segoe UI, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(label, x + nodeWidth / 2, nodeY + 33 * ratio);
      if (index < 2) {
        const arrowX = x + nodeWidth;
        ctx.strokeStyle = "#287dad";
        ctx.beginPath();
        ctx.moveTo(arrowX + 4 * ratio, nodeY + nodeHeight / 2);
        ctx.lineTo(arrowX + gap - 4 * ratio, nodeY + nodeHeight / 2);
        ctx.lineTo(arrowX + gap - 9 * ratio, nodeY + nodeHeight / 2 - 4 * ratio);
        ctx.moveTo(arrowX + gap - 4 * ratio, nodeY + nodeHeight / 2);
        ctx.lineTo(arrowX + gap - 9 * ratio, nodeY + nodeHeight / 2 + 4 * ratio);
        ctx.stroke();
      }
    });
    ctx.textAlign = "left";
    ctx.fillStyle = "#6a7a84";
    ctx.font = `${9 * ratio}px Cascadia Mono, Consolas, monospace`;
    ctx.fillText(state.language === "zh" ? "0 个梯度更新参数 · RGB 运行时定位 · 最多一次重定位" : "0 UPDATED PARAMETERS · RGB RUNTIME POSITION · ONE RETRY MAX", pad.left, height - 18 * ratio);
    return;
  }
  const metrics = (state.adaptation?.events || []).filter(event => event.kind === "metric");
  const values = metrics.flatMap(row => [row.train, row.val]).filter(Number.isFinite);
  const maxY = values.length ? Math.max(...values) * 1.08 : 1;
  const minY = values.length ? Math.max(0, Math.min(...values) * .9) : 0;
  ctx.strokeStyle = "#dde4e8";
  ctx.lineWidth = ratio;
  ctx.fillStyle = "#6a7a84";
  ctx.font = `${10 * ratio}px Cascadia Mono, Consolas, monospace`;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    const label = (maxY - (maxY - minY) * i / 4).toFixed(maxY < .1 ? 4 : 2);
    ctx.fillText(label, 4 * ratio, y + 4 * ratio);
  }
  const maxEpoch = Math.max(1, state.training?.epochs || 1);
  [1, Math.ceil(maxEpoch / 2), maxEpoch].forEach(epoch => {
    const x = pad.left + plotW * (epoch - 1) / Math.max(1, maxEpoch - 1);
    ctx.fillText(String(epoch), x - 4 * ratio, height - 12 * ratio);
  });
  if (!metrics.length) {
    ctx.fillStyle = "#8a989f";
    ctx.font = `${13 * ratio}px Segoe UI, sans-serif`;
    ctx.fillText(state.language === "zh" ? "启动训练后，真实损失曲线会显示在这里。" : "Start training to stream the real loss curve.", pad.left + 20 * ratio, pad.top + plotH / 2);
    return;
  }
  const drawSeries = (key, colour) => {
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2.5 * ratio;
    ctx.beginPath();
    metrics.forEach((row, index) => {
      const x = pad.left + plotW * (row.epoch - 1) / Math.max(1, maxEpoch - 1);
      const y = pad.top + plotH * (maxY - row[key]) / Math.max(1e-12, maxY - minY);
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    metrics.forEach(row => {
      const x = pad.left + plotW * (row.epoch - 1) / Math.max(1, maxEpoch - 1);
      const y = pad.top + plotH * (maxY - row[key]) / Math.max(1e-12, maxY - minY);
      ctx.fillStyle = colour; ctx.beginPath(); ctx.arc(x, y, 3 * ratio, 0, Math.PI * 2); ctx.fill();
    });
  };
  drawSeries("train", "#008f78");
  drawSeries("val", "#287db3");
}

async function refreshFrame() {
  if (state.frameBusy || !state.simulation || !["running", "paused", "completed"].includes(state.simulation.status)) return;
  if (state.simulation.frame_seq === state.frameSequence) return;
  state.frameBusy = true;
  try {
    const response = await fetch(`/api/sim/frame.png?seq=${state.simulation.frame_seq}`, { cache: "no-store" });
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const image = $("simFrame");
    const previous = image.dataset.objectUrl;
    image.onload = () => { if (previous) URL.revokeObjectURL(previous); };
    image.dataset.objectUrl = url;
    image.src = url;
    if (state.activeView === "benchmark") $("benchmarkFrame").src = `/api/sim/frame.png?seq=${state.simulation.frame_seq}&surface=benchmark`;
    state.frameSequence = state.simulation.frame_seq;
  } catch (_) {
    // Status polling handles connection errors.
  } finally {
    state.frameBusy = false;
  }
}

function simulationRequest() {
  return {
    task: $("taskSelect").value,
    policy: $("policySelect").value,
    complexity: $("complexitySelect").value,
    seed: Number($("seedInput").value),
    speed: Number($("speedInput").value)
  };
}

async function startSimulation() {
  try {
    state.simulation = await api("/api/sim/start", { method: "POST", body: JSON.stringify(simulationRequest()) });
    renderStatus();
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

async function sendCommand(command = $("commandInput").value) {
  $("commandFeedback").className = "command-feedback";
  try {
    state.simulation = await api("/api/sim/command", { method: "POST", body: JSON.stringify({ command, ...simulationRequest() }) });
    if (state.config?.tasks[state.simulation.task]) $("taskSelect").value = state.simulation.task;
    if (["medium", "hard", "language"].includes(state.simulation.complexity)) $("complexitySelect").value = state.simulation.complexity;
    if (["rgb_grounded", "structured_state"].includes(state.simulation.policy)) $("policySelect").value = state.simulation.policy;
    $("commandFeedback").textContent = state.language === "zh" ? "指令已解析并发送到实时会话。" : "Command parsed and sent to the live session.";
    renderStatus();
  } catch (error) {
    $("commandFeedback").textContent = error.message.includes("recognised") ? t("commandUnknown") : `${t("requestFailed")}: ${error.message}`;
    $("commandFeedback").className = "command-feedback error";
  }
}

async function controlSimulation(action) {
  try {
    state.simulation = await api("/api/sim/control", { method: "POST", body: JSON.stringify({ action }) });
    renderStatus();
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

function benchmarkRequest() {
  return {
    tasks: [...document.querySelectorAll(".task-checks:not(.policy-checks) input:checked")].map(input => input.value),
    policies: [...document.querySelectorAll(".policy-checks input:checked")].map(input => input.value),
    seed_start: Number($("benchmarkSeed").value),
    seeds_per_task: Number($("benchmarkCount").value),
    speed: Number($("benchmarkSpeed").value),
  };
}

async function startBenchmark() {
  try {
    state.benchmark = await api("/api/benchmark/start", { method: "POST", body: JSON.stringify(benchmarkRequest()) });
    renderBenchmark();
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
}

async function initialise() {
  try {
    const [config, status, studies, portfolio, releases] = await Promise.all([api("/api/config"), api("/api/status"), api("/api/studies"), api("/api/portfolio"), api("/api/releases")]);
    state.config = config;
    state.simulation = status.simulation;
    state.training = status.training;
    state.adaptation = status.adaptation;
    state.benchmark = status.benchmark;
    state.analytics = status.analytics;
    state.studies = studies.studies || [];
    state.governanceSummary = studies.summary;
    state.selectedStudy = state.studies[0] || null;
    state.portfolio = portfolio;
    state.selectedClaimId = portfolio.claims?.some(claim => claim.id === state.selectedClaimId) ? state.selectedClaimId : portfolio.claims?.[0]?.id || null;
    state.releaseData = releases;
    state.selectedReleaseId = releases.releases?.some(release => release.release_id === state.selectedReleaseId) ? state.selectedReleaseId : releases.releases?.[0]?.release_id || null;
    $("connectionLight").className = "status-light online";
    $("connectionLabel").textContent = t("connected");
    renderTaskOptions();
    renderDatasetOptions();
    setLanguage(state.language);
    switchView(state.activeView);
    updateStudyWorkload();
    const events = new EventSource("/api/events");
    events.onmessage = event => {
      const payload = JSON.parse(event.data);
      const previousRunTotal = state.analytics?.total_runs;
      const previousBenchmarkStatus = state.benchmark?.status;
      const previousBenchmarkId = state.benchmark?.benchmark_id;
      state.simulation = payload.simulation;
      state.training = payload.training;
      state.adaptation = payload.adaptation;
      state.benchmark = payload.benchmark;
      state.analytics = payload.analytics;
      $("connectionLight").className = "status-light online";
      $("connectionLabel").textContent = t("connected");
      renderStatus();
      refreshFrame();
      if (state.activeView === "governance" && (previousBenchmarkStatus !== state.benchmark.status || previousBenchmarkId !== state.benchmark.benchmark_id)) refreshStudies();
      if (state.activeView === "experiments" && previousRunTotal !== undefined && previousRunTotal !== state.analytics.total_runs) refreshResearchRecords();
    };
    events.onerror = () => {
      $("connectionLight").className = "status-light offline";
      $("connectionLabel").textContent = t("disconnected");
    };
  } catch (error) {
    $("connectionLight").className = "status-light offline";
    $("connectionLabel").textContent = t("disconnected");
    showToast(`${t("requestFailed")}: ${error.message}`, true);
  }
}

document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => switchView(button.dataset.view)));
$("languageSwitch").addEventListener("click", () => setLanguage(state.language === "zh" ? "en" : "zh"));
$("legacyFrame").addEventListener("load", syncLegacyLanguage);
$("speedInput").addEventListener("input", event => { $("speedValue").textContent = `${Number(event.target.value).toFixed(2)}×`; });
$("taskSelect").addEventListener("change", event => { const suggested = state.config?.tasks[event.target.value]?.complexity; if (suggested) $("complexitySelect").value = suggested; });
$("startSimulation").addEventListener("click", startSimulation);
$("sendCommand").addEventListener("click", () => sendCommand());
$("commandInput").addEventListener("keydown", event => { if (event.key === "Enter") sendCommand(); });
document.querySelectorAll("[data-command-en]").forEach(button => button.addEventListener("click", () => { $("commandInput").value = button.dataset.command; sendCommand(button.dataset.command); }));
document.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => controlSimulation(button.dataset.action)));
$("openNativeViewer").addEventListener("click", async () => {
  try { await api("/api/sim/native-viewer", { method: "POST", body: JSON.stringify(simulationRequest()) }); showToast(t("viewerStarted")); }
  catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
});
$("adaptSource").addEventListener("change", updateAdaptationInstruction);
$("adaptTarget").addEventListener("change", updateAdaptationInstruction);
$("registerAdaptTask").addEventListener("click", registerAdaptationTask);
$("adaptTaskSelect").addEventListener("change", () => { state.adaptationEstimate = null; state.arenaEstimate = null; renderAdaptation(); });
[$("adaptMethod"), $("adaptProfile"), $("adaptEpisodes"), $("adaptEpochs"), $("adaptEvaluationEpisodes"), $("adaptSeed"), $("adaptViewer")].forEach(input => input.addEventListener("change", () => { state.adaptationEstimate = null; state.arenaEstimate = null; renderAdaptation(); }));
document.querySelectorAll("#arenaMethods input, #arenaReuseDataset").forEach(input => input.addEventListener("change", () => { state.arenaEstimate = null; renderAdaptation(); }));
$("estimateAdaptation").addEventListener("click", estimateAdaptation);
$("startAdaptation").addEventListener("click", startAdaptation);
$("startArena").addEventListener("click", startArena);
$("stopAdaptation").addEventListener("click", async () => { try { state.adaptation = await api("/api/adaptation/stop", { method: "POST", body: "{}" }); renderAdaptation(); } catch (error) { showToast(error.message, true); } });
$("openAdaptViewer").addEventListener("click", async () => {
  try {
    await api("/api/adaptation/native-viewer", { method: "POST", body: JSON.stringify({ task_id: $("adaptTaskSelect").value, seed: Number($("adaptSeed").value) }) });
    showToast(t("viewerStarted"));
  } catch (error) { showToast(`${t("requestFailed")}: ${error.message}`, true); }
});
$("toggleAdaptLog").addEventListener("click", () => {
  $("trainingLog").hidden = !$("trainingLog").hidden;
  $("toggleAdaptLog").textContent = $("trainingLog").hidden ? t("showLog") : (state.language === "zh" ? "\u9690\u85cf\u8fc7\u7a0b\u65e5\u5fd7" : "Hide process log");
});
$("benchmarkSpeed").addEventListener("input", event => { $("benchmarkSpeedValue").textContent = `${Number(event.target.value).toFixed(2)}×`; });
$("startBenchmark").addEventListener("click", startBenchmark);
$("stopBenchmark").addEventListener("click", async () => { try { state.benchmark = await api("/api/benchmark/stop", { method: "POST", body: "{}" }); renderBenchmark(); } catch (error) { showToast(error.message, true); } });
document.querySelectorAll(".governance-task-checks input").forEach(input => input.addEventListener("change", updateStudyWorkload));
$("studyCountInput").addEventListener("input", updateStudyWorkload);
$("lockProtocol").addEventListener("click", lockProtocol);
$("launchStudy").addEventListener("click", launchStudy);
$("studyList").addEventListener("click", event => {
  const row = event.target.closest("[data-study-id]");
  if (!row) return;
  state.selectedStudy = state.studies.find(study => study.study_id === row.dataset.studyId) || null;
  renderGovernance();
});
$("claimList").addEventListener("click", event => {
  const row = event.target.closest("[data-claim-id]");
  if (!row) return;
  state.selectedClaimId = row.dataset.claimId;
  renderPortfolio();
  syncRoute();
});
$("lifecycleFilters").addEventListener("click", event => {
  const button = event.target.closest("[data-lifecycle]");
  if (!button) return;
  state.selectedLifecycle = button.dataset.lifecycle;
  renderPortfolio();
  syncRoute();
});
$("openClaimStudy").addEventListener("click", () => {
  const studyId = $("openClaimStudy").dataset.studyId;
  state.selectedStudy = state.studies.find(study => study.study_id === studyId) || state.selectedStudy;
  switchView("governance");
  renderGovernance();
});
$("createRelease").addEventListener("click", createEvidenceRelease);
$("releaseList").addEventListener("click", event => {
  const row = event.target.closest("[data-release-id]");
  if (!row) return;
  state.selectedReleaseId = row.dataset.releaseId;
  renderReleases();
  syncRoute();
});
$("runKindFilter").addEventListener("change", refreshResearchRecords);
$("runStatusFilter").addEventListener("change", refreshResearchRecords);
$("runRows").addEventListener("click", event => {
  const rowElement = event.target.closest("tr[data-run-id]");
  if (!rowElement) return;
  state.selectedRun = state.runs.find(row => row.run_id === rowElement.dataset.runId) || null;
  renderRuns();
});
$("copyRunCommand").addEventListener("click", async () => {
  const command = state.selectedRun?.command;
  if (!command) return;
  await navigator.clipboard.writeText(command);
  showToast(t("commandCopied"));
});
window.addEventListener("resize", () => { if (state.activeView === "training") drawLossChart(); });
setLanguage(state.language);
initialise();
