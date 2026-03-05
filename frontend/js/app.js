var API_URL = '/api/predict';
var LIVE_GAME_URL = '/api/live-game';

var RANK_COLORS = {
    IRON: '#6B5B50',
    BRONZE: '#CD7F32',
    SILVER: '#A8B4C0',
    GOLD: '#FFD700',
    PLATINUM: '#00CED1',
    EMERALD: '#50C878',
    DIAMOND: '#B9F2FF'
};

var RANK_ORDER = ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND'];

var charts = {};

function destroyCharts() {
    Object.keys(charts).forEach(function(key) {
        if (charts[key]) charts[key].destroy();
    });
    charts = {};
}

function showLoader(show) {
    document.getElementById('loader').classList.toggle('hidden', !show);
}

function showError(msg) {
    var el = document.getElementById('error');
    el.textContent = msg;
    el.classList.remove('hidden');
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
}

function formatNum(val) {
    if (Math.abs(val) >= 1000) return val.toFixed(0);
    if (Math.abs(val) >= 100) return val.toFixed(1);
    if (Math.abs(val) >= 10) return val.toFixed(1);
    return val.toFixed(2);
}

async function analyze() {
    var input = document.getElementById('riotId').value.trim();

    if (!input) {
        showError('Please enter your Riot ID');
        return;
    }

    if (!input.includes('#')) {
        showError('Enter your Riot ID in format: Name#TAG');
        return;
    }

    var parts = input.split('#');
    var gameName = parts[0].trim();
    var tagLine = parts.slice(1).join('#').trim();

    if (!gameName || !tagLine) {
        showError('Enter your Riot ID in format: Name#TAG');
        return;
    }

    showLoader(true);
    hideError();
    hideResults();
    destroyCharts();

    try {
        var region = document.getElementById('regionSelect').value;
        var response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_name: gameName, tag_line: tagLine, region: region })
        });

        if (!response.ok) {
            var detail = 'Analysis failed';
            try {
                var err = await response.json();
                detail = err.detail || detail;
            } catch (e) {}
            throw new Error(detail);
        }

        var data = await response.json();
        renderResults(data);
    } catch (e) {
        showError(e.message);
    } finally {
        showLoader(false);
    }
}

function renderResults(data) {
    document.getElementById('results').classList.remove('hidden');
    renderSummoner(data.summoner);
    renderRankCard(data.prediction);
    renderWinrateChart(data.record);
    renderProbabilityChart(data.prediction.rank_probabilities);
    renderRadarChart(data.comparison);
    renderComparisonBar(data.comparison, data.prediction.predicted_rank);
    renderComparisonTables(data.comparison, data.prediction.predicted_rank);
    renderMatchHistory(data.matches);
    renderTrendChart(data.matches);
    renderChampionRecommender(data.champion_recommendations);
}

function renderSummoner(summoner) {
    document.getElementById('summonerName').textContent =
        summoner.game_name + '#' + summoner.tag_line;
}

function renderRankCard(prediction) {
    var rank = prediction.predicted_rank;
    var conf = (prediction.confidence * 100).toFixed(1);
    var color = RANK_COLORS[rank] || '#8e9297';

    document.getElementById('rankName').textContent = rank;
    document.getElementById('rankName').style.color = color;
    document.getElementById('confidence').textContent = conf + '% confidence';
    document.getElementById('confidenceFill').style.width = conf + '%';
    document.getElementById('confidenceFill').style.background =
        'linear-gradient(90deg, ' + color + '88, ' + color + ')';

    var icon = document.getElementById('rankIcon');
    icon.textContent = rank.charAt(0);
    icon.style.background = 'linear-gradient(135deg, ' + color + '66, ' + color + ')';
    icon.style.boxShadow = '0 4px 24px ' + color + '33';
}

function renderWinrateChart(record) {
    var ctx = document.getElementById('winrateChart').getContext('2d');
    var wr = record.total > 0 ? ((record.wins / record.total) * 100).toFixed(0) : '0';

    document.getElementById('winrateCenter').textContent = wr + '%';
    document.getElementById('recordText').innerHTML =
        '<span class="wins">' + record.wins + 'W</span> / <span class="losses">' + record.losses + 'L</span>';

    charts.winrate = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Wins', 'Losses'],
            datasets: [{
                data: [record.wins, record.losses],
                backgroundColor: ['#22c55e', '#ef4444'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            cutout: '75%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a1d26',
                    titleColor: '#e8e8e8',
                    bodyColor: '#8e9297',
                    borderColor: '#2a2d3a',
                    borderWidth: 1
                }
            }
        }
    });
}

function renderProbabilityChart(probabilities) {
    var ctx = document.getElementById('probabilityChart').getContext('2d');
    var values = RANK_ORDER.map(function(r) { return (probabilities[r] || 0) * 100; });
    var colors = RANK_ORDER.map(function(r) { return RANK_COLORS[r]; });

    charts.probability = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: RANK_ORDER,
            datasets: [{
                data: values,
                backgroundColor: colors.map(function(c) { return c + '77'; }),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a1d26',
                    borderColor: '#2a2d3a',
                    borderWidth: 1,
                    callbacks: {
                        label: function(ctx) { return ctx.parsed.y.toFixed(1) + '%'; }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: '#2a2d3a44' },
                    ticks: { color: '#8e9297', callback: function(v) { return v + '%'; } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8e9297', font: { weight: '600' } }
                }
            }
        }
    });
}

function renderRadarChart(comparison) {
    var ctx = document.getElementById('radarChart').getContext('2d');
    var metrics = [];
    var percentiles = [];

    Object.keys(comparison).forEach(function(cat) {
        comparison[cat].forEach(function(m) {
            metrics.push(m.label);
            percentiles.push(m.inverted ? 100 - m.percentile : m.percentile);
        });
    });

    charts.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metrics,
            datasets: [{
                label: 'Your Percentile',
                data: percentiles,
                backgroundColor: 'rgba(50, 115, 250, 0.15)',
                borderColor: '#3273fa',
                borderWidth: 2,
                pointBackgroundColor: '#3273fa',
                pointBorderColor: '#1a1d26',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Rank Average (50th)',
                data: new Array(metrics.length).fill(50),
                backgroundColor: 'transparent',
                borderColor: '#5b5e66',
                borderWidth: 1,
                borderDash: [6, 4],
                pointRadius: 0,
                pointHoverRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 25,
                        color: '#5b5e66',
                        backdropColor: 'transparent',
                        font: { size: 9 }
                    },
                    grid: { color: '#2a2d3a88' },
                    pointLabels: {
                        color: '#a8acb2',
                        font: { size: 11, weight: '500' }
                    },
                    angleLines: { color: '#2a2d3a66' }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#8e9297',
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 16,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a1d26',
                    borderColor: '#2a2d3a',
                    borderWidth: 1,
                    callbacks: {
                        label: function(ctx) {
                            return ctx.dataset.label + ': ' + ctx.parsed.r.toFixed(1) + '%';
                        }
                    }
                }
            }
        }
    });
}

function renderComparisonBar(comparison, rank) {
    var ctx = document.getElementById('comparisonChart').getContext('2d');
    var labels = [];
    var diffs = [];
    var barColors = [];

    Object.keys(comparison).forEach(function(cat) {
        comparison[cat].forEach(function(m) {
            labels.push(m.label);
            var dp = m.diff_percent;
            if (m.inverted) dp = -dp;
            diffs.push(dp);
            barColors.push(dp >= 0 ? '#22c55e' : '#ef4444');
        });
    });

    charts.comparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Difference from ' + rank + ' avg',
                data: diffs,
                backgroundColor: barColors.map(function(c) { return c + '55'; }),
                borderColor: barColors,
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1a1d26',
                    borderColor: '#2a2d3a',
                    borderWidth: 1,
                    callbacks: {
                        label: function(ctx) {
                            var v = ctx.parsed.x;
                            return (v > 0 ? '+' : '') + v.toFixed(1) + '% vs avg';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#2a2d3a44' },
                    ticks: {
                        color: '#8e9297',
                        callback: function(v) { return (v > 0 ? '+' : '') + v + '%'; }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#a8acb2', font: { size: 11, weight: '500' } }
                }
            }
        }
    });
}

function renderComparisonTables(comparison, rank) {
    document.getElementById('comparedRank').textContent = rank;
    document.getElementById('comparedRank').style.color = RANK_COLORS[rank] || '#e8e8e8';
    var container = document.getElementById('comparisonTables');
    container.innerHTML = '';

    var categoryNames = {
        combat: 'Combat',
        farming: 'Farming',
        vision: 'Vision',
        survivability: 'Survivability',
        economy: 'Economy'
    };

    Object.keys(comparison).forEach(function(key) {
        var metrics = comparison[key];
        var div = document.createElement('div');
        div.className = 'comparison-category';

        var html = '<div class="category-header">' + (categoryNames[key] || key) + '</div>';
        html += '<div class="metric-row header-row">';
        html += '<span>Metric</span>';
        html += '<span style="text-align:right">You</span>';
        html += '<span style="text-align:right">Avg</span>';
        html += '<span style="text-align:right">Diff</span>';
        html += '<span style="text-align:right">Pctl</span>';
        html += '</div>';

        metrics.forEach(function(m) {
            var diffClass = m.above_average ? 'above' : 'below';
            var diffSign = m.diff_absolute >= 0 ? '+' : '';
            var pctile = m.inverted ? (100 - m.percentile).toFixed(0) : m.percentile.toFixed(0);
            var pctClass = parseFloat(pctile) >= 50 ? 'above' : 'below';
            var pctBg = pctClass === 'above' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)';

            html += '<div class="metric-row">';
            html += '<span class="metric-label">' + m.label + '</span>';
            html += '<span class="metric-value">' + formatNum(m.player_value) + '</span>';
            html += '<span class="metric-value">' + formatNum(m.rank_average) + '</span>';
            html += '<span class="metric-value ' + diffClass + '">' + diffSign + m.diff_percent.toFixed(1) + '%</span>';
            html += '<span class="percentile-badge ' + pctClass + '" style="background:' + pctBg + '">' + pctile + '</span>';
            html += '</div>';
        });

        div.innerHTML = html;
        container.appendChild(div);
    });
}

function renderMatchHistory(matches) {
    var container = document.getElementById('matchesGrid');
    container.innerHTML = '';

    matches.forEach(function(m) {
        var isWin = m.win === 1;
        var div = document.createElement('div');
        div.className = 'match-card ' + (isWin ? 'win' : 'loss');

        div.innerHTML =
            '<div class="match-top">' +
                '<div>' +
                    '<div class="match-champion">' + m.champion + '</div>' +
                    '<div class="match-role">' + m.role + '</div>' +
                '</div>' +
                '<span class="match-result ' + (isWin ? 'win' : 'loss') + '">' +
                    (isWin ? 'WIN' : 'LOSS') +
                '</span>' +
            '</div>' +
            '<div class="match-stats">' +
                '<div class="match-stat">' +
                    '<div class="match-stat-value">' + m.kills + '/' + m.deaths + '/' + m.assists + '</div>' +
                    '<div class="match-stat-label">KDA</div>' +
                '</div>' +
                '<div class="match-stat">' +
                    '<div class="match-stat-value">' + m.cs_per_min.toFixed(1) + '</div>' +
                    '<div class="match-stat-label">CS/min</div>' +
                '</div>' +
                '<div class="match-stat">' +
                    '<div class="match-stat-value">' + m.damage_per_min.toFixed(0) + '</div>' +
                    '<div class="match-stat-label">DMG/min</div>' +
                '</div>' +
            '</div>';

        container.appendChild(div);
    });
}

function renderTrendChart(matches) {
    var ctx = document.getElementById('trendChart').getContext('2d');
    var labels = matches.map(function(_, i) { return 'G' + (i + 1); });
    var kdas = matches.map(function(m) { return m.kda; });
    var cs = matches.map(function(m) { return m.cs_per_min; });

    var pointColorsKda = matches.map(function(m) {
        return m.win ? '#22c55e' : '#ef4444';
    });

    charts.trend = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'KDA',
                data: kdas,
                borderColor: '#3273fa',
                backgroundColor: 'rgba(50, 115, 250, 0.08)',
                fill: true,
                tension: 0.35,
                pointBackgroundColor: pointColorsKda,
                pointBorderColor: pointColorsKda,
                pointRadius: 6,
                pointHoverRadius: 8,
                borderWidth: 2.5,
                yAxisID: 'y'
            }, {
                label: 'CS/min',
                data: cs,
                borderColor: '#f59e0b',
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.35,
                borderDash: [6, 4],
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#f59e0b',
                borderWidth: 2,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#8e9297',
                        usePointStyle: true,
                        padding: 16,
                        font: { size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: '#1a1d26',
                    borderColor: '#2a2d3a',
                    borderWidth: 1,
                    callbacks: {
                        afterLabel: function(ctx) {
                            if (ctx.datasetIndex === 0) {
                                var m = matches[ctx.dataIndex];
                                return m.champion + ' (' + (m.win ? 'Win' : 'Loss') + ')';
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    grid: { color: '#2a2d3a44' },
                    ticks: { color: '#3273fa', font: { size: 10 } },
                    title: { display: true, text: 'KDA', color: '#3273fa', font: { size: 11 } }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#f59e0b', font: { size: 10 } },
                    title: { display: true, text: 'CS/min', color: '#f59e0b', font: { size: 11 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8e9297', font: { size: 10 } }
                }
            }
        }
    });
}

function getChampImageUrl(champName) {
    return 'https://ddragon.leagueoflegends.com/cdn/14.10.1/img/champion/' + champName + '.png';
}

function renderChampionRecommender(champData) {
    var container = document.getElementById('recommenderContent');
    if (!champData) {
        container.innerHTML = '<p style="color:var(--text-muted)">No champion data available</p>';
        return;
    }

    var html = '';

    // Player's champion pool
    var pool = champData.player_champions || [];
    if (pool.length > 0) {
        html += '<div class="champ-pool-label">YOUR CHAMPION POOL</div>';
        html += '<div class="champ-pool">';
        pool.forEach(function(c) {
            var wrClass = c.winrate >= 50 ? 'above' : 'below';
            var imgUrl = getChampImageUrl(c.champion);
            html += '<div class="champ-pool-item">';
            html += '<img class="champ-pool-img" src="' + escapeAttr(imgUrl) + '" alt="' + escapeAttr(c.champion) + '" onerror="this.style.visibility=\'hidden\'">';
            html += '<div class="champ-pool-info">';
            html += '<div class="champ-pool-name">' + escapeHtml(c.champion) + '</div>';
            html += '<div class="champ-pool-stats">';
            html += '<span class="' + wrClass + '">' + c.winrate + '% WR</span>';
            html += '<span class="champ-games">' + c.games + ' games</span>';
            html += '</div></div></div>';
        });
        html += '</div>';
    }

    // Recommendations
    var recs = champData.recommendations || [];
    if (recs.length > 0) {
        html += '<div class="champ-pool-label rec-label">RECOMMENDED FOR YOU</div>';
        html += '<p class="rec-desc">Based on your playstyle (cosine similarity with 40k+ ranked games dataset)</p>';
        html += '<div class="champ-recs">';
        recs.forEach(function(r, i) {
            var imgUrl = getChampImageUrl(r.champion);
            html += '<div class="champ-rec-card">';
            html += '<div class="rec-rank">#' + (i + 1) + '</div>';
            html += '<img class="champ-rec-img" src="' + escapeAttr(imgUrl) + '" alt="' + escapeAttr(r.champion) + '" onerror="this.style.visibility=\'hidden\'">';
            html += '<div class="champ-rec-name">' + escapeHtml(r.champion) + '</div>';
            html += '<div class="champ-rec-stats">';
            html += '<div class="rec-stat"><span class="rec-stat-val">' + r.match_score + '%</span><span class="rec-stat-lbl">Match</span></div>';
            html += '<div class="rec-stat"><span class="rec-stat-val">' + r.dataset_winrate + '%</span><span class="rec-stat-lbl">Avg WR</span></div>';
            html += '<div class="rec-stat"><span class="rec-stat-val">' + r.dataset_games + '</span><span class="rec-stat-lbl">Games</span></div>';
            html += '</div></div>';
        });
        html += '</div>';
    }

    container.innerHTML = html;
}

document.getElementById('analyzeBtn').addEventListener('click', analyze);

document.getElementById('riotId').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') analyze();
});

document.getElementById('liveGameBtn').addEventListener('click', checkLiveGame);

async function checkLiveGame() {
    var input = document.getElementById('riotId').value.trim();
    if (!input || !input.includes('#')) {
        showError('Enter a Riot ID first (e.g. Player#EUW)');
        return;
    }

    var parts = input.split('#');
    var gameName = parts[0].trim();
    var tagLine = parts.slice(1).join('#').trim();
    if (!gameName || !tagLine) return;

    var btn = document.getElementById('liveGameBtn');
    var loader = document.getElementById('liveGameLoader');
    var status = document.getElementById('liveGameStatus');
    var content = document.getElementById('liveGameContent');

    btn.disabled = true;
    btn.textContent = 'CHECKING...';
    loader.classList.remove('hidden');
    status.classList.add('hidden');
    content.classList.add('hidden');
    content.innerHTML = '';

    try {
        var region = document.getElementById('regionSelect').value;
        var response = await fetch(LIVE_GAME_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_name: gameName, tag_line: tagLine, region: region })
        });

        if (!response.ok) {
            var err = {};
            try { err = await response.json(); } catch (e) {}
            throw new Error(err.detail || 'Failed to check live game');
        }

        var data = await response.json();

        if (!data.in_game) {
            status.textContent = 'Player is not currently in a game';
            status.className = 'live-game-status not-in-game';
            status.classList.remove('hidden');
        } else {
            renderLiveGame(data);
        }
    } catch (e) {
        status.textContent = e.message;
        status.className = 'live-game-status live-error';
        status.classList.remove('hidden');
    } finally {
        loader.classList.add('hidden');
        btn.disabled = false;
        btn.textContent = 'CHECK LIVE GAME';
    }
}

function renderLiveGame(data) {
    var content = document.getElementById('liveGameContent');
    content.classList.remove('hidden');

    var html = '<div class="live-teams">';

    html += renderTeamColumn(data.blue_team, 'blue', 'BLUE TEAM');
    html += renderTeamColumn(data.red_team, 'red', 'RED TEAM');

    html += '</div>';

    var pred = data.prediction;
    var winner = pred.predicted_winner;
    html += '<div class="win-prediction">';
    html += '<div class="prediction-header">WIN PREDICTION</div>';
    html += '<div class="prediction-bar">';
    html += '<div class="prediction-blue' + (winner === 'blue' ? ' winner' : '') + '" style="width:' + pred.blue_win_probability + '%">';
    html += 'BLUE ' + pred.blue_win_probability + '%';
    html += '</div>';
    html += '<div class="prediction-red' + (winner === 'red' ? ' winner' : '') + '" style="width:' + pred.red_win_probability + '%">';
    html += 'RED ' + pred.red_win_probability + '%';
    html += '</div>';
    html += '</div></div>';

    content.innerHTML = html;
}

function renderTeamColumn(team, side, title) {
    var html = '<div class="live-team">';
    html += '<div class="team-header ' + side + '">' + title + '</div>';

    team.forEach(function(p) {
        var rankColor = RANK_COLORS[p.predicted_rank] || '#5b5e66';
        var isSearched = p.is_searched_player ? ' is-searched' : '';

        html += '<div class="live-player' + isSearched + '">';
        html += '<img class="champ-img" src="' + escapeAttr(p.champion_image) + '" alt="' + escapeAttr(p.champion_name) + '" onerror="this.style.display=\'none\'">';
        html += '<div class="player-info">';
        html += '<div class="player-name">' + escapeHtml(p.summoner_name);
        if (p.summoner_tag) html += '<span class="player-tag">#' + escapeHtml(p.summoner_tag) + '</span>';
        html += '</div>';
        html += '<div class="player-champ">' + escapeHtml(p.champion_name) + '</div>';
        html += '</div>';
        html += '<div class="player-stats-live">';

        if (p.winrate !== null && p.winrate !== undefined) {
            var wrClass = p.winrate >= 50 ? 'above' : 'below';
            html += '<span class="player-wr ' + wrClass + '">' + p.winrate + '% WR</span>';
        } else {
            html += '<span class="player-wr dim">N/A</span>';
        }

        if (p.predicted_rank) {
            html += '<span class="player-rank-badge" style="background:' + rankColor + '22;color:' + rankColor + ';border:1px solid ' + rankColor + '44">';
            html += p.predicted_rank;
            html += '</span>';
        } else {
            html += '<span class="player-rank-badge dim-badge">N/A</span>';
        }

        html += '</div></div>';
    });

    html += '</div>';
    return html;
}

function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
