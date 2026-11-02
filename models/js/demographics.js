/**
 * demographics.js — Demographic Risk Determinants Engine v1
 *
 * Analyses a user's demographic profile (age, gender, occupation, area)
 * against India-specific epidemiological data to produce:
 *   • A population prevalence estimate for their group
 *   • Evidence-based risk/protective factors
 *   • A comparison insight sentence
 *   • A risk multiplier that modulates the composite score
 *
 * Data sources:
 *   - NIMHANS National Mental Health Survey 2016 (NMHS-2016)
 *   - Lancet India, Patel et al. — urban/rural gap
 *   - WHO South-East Asia Region — gender × depression
 *   - AIIMS/NIMHANS student cohort studies
 *   - India Suicide Report 2022, NCRB
 *   - Various peer-reviewed Indian occupational health studies
 *
 * NOTE: All prevalence figures are population-level estimates.
 *       This is NOT a clinical diagnosis tool.
 */
const Demographics = (function () {

  // ── Age-group base prevalence (% of group with clinically significant
  //    depression/anxiety in India, NMHS-2016 + updated estimates) ──────────
  const AGE_DATA = {
    'under18': {
      label: 'Under 18 (Adolescent)',
      prevalence: 17,
      riskWeight: 1.25,
      riskFactors: [
        'Academic pressure and exam stress',
        'Peer rejection and social comparison (social media)',
        'Limited emotional vocabulary and coping skills',
        'Hormonal changes amplify emotional reactivity'
      ],
      protective: ['Strong family bonds', 'School counsellor access', 'Sports and physical activity'],
      note: '1 in 6 Indian adolescents meets criteria for a mental health disorder (WHO 2021)'
    },
    '18-24': {
      label: '18–24 (Young Adult)',
      prevalence: 29,
      riskWeight: 1.40,
      riskFactors: [
        'College/university stress, competitive exams (JEE/NEET/UPSC)',
        'Social isolation and identity uncertainty',
        'Financial dependence combined with adult responsibilities',
        'First exposure to major life transitions without adequate support'
      ],
      protective: ['Peer support networks', 'Physical activity', 'Creative outlets'],
      note: '~29% of Indian college students show significant depressive symptoms (AIIMS/NIMHANS studies)'
    },
    '25-34': {
      label: '25–34 (Early Career)',
      prevalence: 19,
      riskWeight: 1.15,
      riskFactors: [
        'Workplace stress and job insecurity',
        'Relationship and marriage pressure',
        'Career vs. personal identity conflict',
        'Financial obligations (loans, family support)'
      ],
      protective: ['Growing autonomy', 'Career growth opportunities', 'Building support systems'],
      note: '~19% in this group report moderate-to-severe anxiety (NMHS 2016)'
    },
    '35-44': {
      label: '35–44 (Mid-Career)',
      prevalence: 16,
      riskWeight: 1.05,
      riskFactors: [
        'Burnout from prolonged occupational stress',
        'Dual burden: career + family caregiving',
        'Mid-life identity and purpose questioning',
        'Chronic health conditions beginning to emerge'
      ],
      protective: ['Established support systems', 'Career stability', 'Clearer sense of identity'],
      note: 'Mid-life is associated with lower self-reported well-being globally (Blanchflower & Oswald)'
    },
    '45-59': {
      label: '45–59 (Pre-Retirement)',
      prevalence: 18,
      riskWeight: 1.10,
      riskFactors: [
        'Empty nest syndrome and changing family roles',
        'Caring for aging parents (sandwich generation)',
        'Menopause-related mood changes (for women)',
        'Increased chronic disease burden'
      ],
      protective: ['Wisdom and life experience', 'Financial stability (often)', 'Stronger coping strategies'],
      note: 'Depression risk rises again after 45, especially for caregivers (NMHS 2016)'
    },
    '60plus': {
      label: '60+ (Older Adult)',
      prevalence: 24,
      riskWeight: 1.30,
      riskFactors: [
        'Social isolation and loneliness',
        'Loss of purpose post-retirement',
        'Chronic illness and functional decline',
        'Bereavement (loss of peers and spouse)',
        'Digital exclusion — reduced access to mental health resources'
      ],
      protective: ['Spiritual practices', 'Community ties', 'Grandparenting role'],
      note: '~24% of Indian adults 60+ show clinically significant depressive symptoms (Geriatric studies, India)'
    }
  };

  // ── Gender modifiers ──────────────────────────────────────────────────────
  const GENDER_DATA = {
    'female': {
      label: 'Female',
      multiplier: 1.20,
      insight: 'Women in India are 1.5× more likely to experience depression than men (NMHS 2016), partly due to gender-based stressors including domestic burden, limited autonomy, and underreporting stigma.',
      riskFactors: ['Domestic violence exposure', 'Double burden (work + home)', 'Reproductive mental health challenges', 'Social pressure around marriage and children'],
      protective: ['Stronger social support seeking', 'Higher emotional intelligence', 'Family connectedness']
    },
    'male': {
      label: 'Male',
      multiplier: 1.00,
      insight: 'Men in India are significantly less likely to seek help for mental health issues, leading to underdiagnosis. Indian men account for ~75% of suicide deaths (NCRB 2022).',
      riskFactors: ['Social pressure to suppress emotions ("be strong")', 'Higher substance use as coping', 'Less likely to seek professional help', 'Provider-role pressure'],
      protective: ['Physical activity culture', 'Peer group bonding (sport, work)']
    },
    'nonbinary': {
      label: 'Non-binary / Other',
      multiplier: 1.35,
      insight: 'Gender minority individuals face significantly elevated mental health risks due to social stigma, discrimination, and lack of affirming support systems in India.',
      riskFactors: ['Social stigma and discrimination', 'Family rejection', 'Limited access to affirming mental healthcare', 'Identity concealment stress'],
      protective: ['Community and peer support', 'Online support networks', 'Strong personal identity']
    },
    'prefer_not': {
      label: 'Prefer not to say',
      multiplier: 1.00,
      insight: null,
      riskFactors: [],
      protective: []
    }
  };

  // ── Occupation risk profiles ───────────────────────────────────────────────
  const OCCUPATION_DATA = {
    'student': {
      label: 'Student',
      prevalence: 29,
      multiplier: 1.35,
      emoji: '🎓',
      riskFactors: [
        'Exam pressure and academic competition (JEE/NEET/board exams)',
        'Uncertainty about future (career, placements, admissions)',
        'Distance from family and hometown',
        'Financial stress (fees, hostel, living expenses)',
        'Social comparison and peer pressure'
      ],
      protective: ['Campus counselling services', 'Peer learning communities', 'Youth resilience'],
      note: 'Students are the highest-risk occupational group in India for anxiety/depression'
    },
    'school_teacher': {
      label: 'School Teacher',
      prevalence: 22,
      multiplier: 1.20,
      emoji: '📚',
      riskFactors: [
        'Emotional labour — managing children\'s stress daily',
        'Administrative burden and paperwork overload',
        'Low salary relative to qualifications',
        'Parental expectations and pressure',
        'Limited career advancement'
      ],
      protective: ['Meaningful work and purpose', 'School community bonds', 'Summer/winter breaks'],
      note: '~22% of Indian school teachers report occupational burnout (teacher well-being studies)'
    },
    'college_faculty': {
      label: 'College Faculty',
      prevalence: 24,
      multiplier: 1.25,
      emoji: '🏫',
      riskFactors: [
        'Research pressure (publish or perish culture)',
        'Administrative overload alongside teaching',
        'Contractual/guest lecturer insecurity',
        'Peer comparison in academic ranking',
        'Student mental health responsibility without training'
      ],
      protective: ['Academic community', 'Intellectual stimulation', 'Teaching impact'],
      note: 'Indian academia has high burnout rates, especially for contractual and junior faculty'
    },
    'healthcare': {
      label: 'Healthcare Worker',
      prevalence: 35,
      multiplier: 1.45,
      emoji: '🏥',
      riskFactors: [
        'Compassion fatigue from patient suffering',
        'Long shifts and irregular hours',
        'High-stakes decision making',
        'Post-COVID burnout and trauma',
        'Workplace violence in Indian healthcare settings'
      ],
      protective: ['Strong professional identity', 'Team support', 'Structured coping protocols'],
      note: 'Healthcare workers have the highest occupational depression prevalence — ~35% in India post-COVID'
    },
    'it_tech': {
      label: 'IT / Tech Worker',
      prevalence: 25,
      multiplier: 1.25,
      emoji: '💻',
      riskFactors: [
        'Always-on work culture and late-night calls across time zones',
        'Job insecurity in recent tech layoff cycles',
        'Sedentary lifestyle and screen fatigue',
        'Social isolation (WFH) and loneliness',
        'Imposter syndrome in high-performance environments'
      ],
      protective: ['Financial stability', 'Remote flexibility', 'Strong peer networks'],
      note: '~25% of Indian IT professionals report significant stress/anxiety (industry surveys)'
    },
    'business': {
      label: 'Business / Self-employed',
      prevalence: 20,
      multiplier: 1.15,
      emoji: '💼',
      riskFactors: [
        'Financial uncertainty and business risk',
        'No work-life boundary',
        'Isolation of entrepreneurship',
        'Pressure to appear successful'
      ],
      protective: ['Autonomy and control', 'Strong purpose', 'Community of entrepreneurs'],
      note: 'Entrepreneurs face unique risk from isolation and financial volatility'
    },
    'government': {
      label: 'Government / PSU Employee',
      prevalence: 13,
      multiplier: 0.90,
      emoji: '🏛️',
      riskFactors: [
        'Bureaucratic frustration and slow career growth',
        'Transfer stress (frequent relocations)',
        'Rigid hierarchy and limited autonomy'
      ],
      protective: ['Job security (lowest occupational stressor)', 'Pension and benefits', 'Regular hours'],
      note: 'Government employees have relatively lower mental health risk due to job security'
    },
    'homemaker': {
      label: 'Homemaker',
      prevalence: 23,
      multiplier: 1.20,
      emoji: '🏠',
      riskFactors: [
        'Social isolation and lack of adult interaction',
        'Invisible labour not valued economically',
        'Loss of professional identity',
        'Financial dependence and limited autonomy',
        'Unrecognised burnout'
      ],
      protective: ['Family bonds', 'Community with other homemakers', 'Flexible daily schedule'],
      note: 'Homemakers — especially women — face significant depression risk due to isolation and invisible labour'
    },
    'unemployed': {
      label: 'Unemployed / Job Seeking',
      prevalence: 38,
      multiplier: 1.55,
      emoji: '🔍',
      riskFactors: [
        'Loss of identity and purpose tied to work',
        'Financial stress and uncertainty',
        'Social stigma around unemployment in India',
        'Daily routine disruption',
        'Prolonged rejection from applications'
      ],
      protective: ['Community support', 'Skill-building activities', 'Volunteer work'],
      note: 'Unemployment doubles depression risk — among the highest-risk occupational categories'
    },
    'other': {
      label: 'Other',
      prevalence: 18,
      multiplier: 1.00,
      emoji: '💡',
      riskFactors: ['Occupational stress varies by role'],
      protective: ['Adaptability', 'Diverse skill set'],
      note: 'Occupational mental health risk varies widely — contextual factors matter most'
    }
  };

  // ── Area modifiers ────────────────────────────────────────────────────────
  const AREA_DATA = {
    'urban_metro': {
      label: 'Urban Metro',
      multiplier: 1.20,
      insight: 'Urban metro residents in India face higher reported rates of anxiety and depression (20–30%), driven by traffic, noise, work pressure, and social isolation despite being in crowds.',
      riskFactors: ['High cost of living', 'Commute stress', 'Social fragmentation', 'Competitive pressure'],
      protective: ['Better healthcare access', 'More awareness of mental health', 'More support resources available']
    },
    'urban_town': {
      label: 'Urban Town',
      multiplier: 1.08,
      insight: 'Urban town residents experience moderate urban stress with some community protective factors from smaller, more connected social networks.',
      riskFactors: ['Aspiration-reality gap', 'Limited mental health resources compared to metros'],
      protective: ['Community bonds', 'Lower cost of living', 'Slower pace than metros']
    },
    'semi_rural': {
      label: 'Semi-rural',
      multiplier: 0.95,
      insight: 'Semi-rural areas offer community protective factors but have significant gaps in mental health access and awareness.',
      riskFactors: ['Limited mental health services', 'Agricultural distress (if farming-related)', 'Lower mental health literacy'],
      protective: ['Strong community and family bonds', 'Lower social comparison pressure', 'Nature access']
    },
    'rural': {
      label: 'Rural',
      multiplier: 0.90,
      insight: 'Rural India has lower reported prevalence (10–15%) but critically lower access to diagnosis and treatment, meaning actual rates are likely higher than reported.',
      riskFactors: ['Severely limited mental health access', 'Agricultural distress and debt', 'Stigma around help-seeking', 'Fewer formal support systems'],
      protective: ['Extended family support', 'Community cohesion', 'Lower social competition'],
      note: 'Rural mental health is significantly under-detected — access, not prevalence, is the core issue (Lancet India)'
    }
  };

  // ── Main analysis function ────────────────────────────────────────────────
  function analyze(profile) {
    const { age, gender, occupation, area } = profile || {};

    const hasData = age || gender || occupation || area;
    if (!hasData) return null;

    const ageData  = AGE_DATA[age]          || null;
    const genData  = GENDER_DATA[gender]    || null;
    const occData  = OCCUPATION_DATA[occupation] || null;
    const areaData = AREA_DATA[area]        || null;

    // ── Base prevalence ────────────────────────────────────────────────────
    // Start from occupation prevalence (most impactful), fall back to age
    let prevalence = occData?.prevalence || ageData?.prevalence || 18;

    // ── Risk multiplier ────────────────────────────────────────────────────
    // Combine all active multipliers (product, capped at 1.8)
    let multiplier = 1.0;
    if (ageData)  multiplier *= ageData.riskWeight  || 1.0;
    if (genData)  multiplier *= genData.multiplier  || 1.0;
    if (areaData) multiplier *= areaData.multiplier || 1.0;
    multiplier = Math.min(1.8, Math.max(0.7, multiplier));

    // ── Risk label ─────────────────────────────────────────────────────────
    let atRiskLabel, atRiskColor;
    if (multiplier >= 1.40) {
      atRiskLabel = 'High-Risk Group';    atRiskColor = '#ef4444';
    } else if (multiplier >= 1.15) {
      atRiskLabel = 'Elevated-Risk Group'; atRiskColor = '#f59e0b';
    } else if (multiplier <= 0.95) {
      atRiskLabel = 'Lower-Risk Group';   atRiskColor = '#10b981';
    } else {
      atRiskLabel = 'Moderate-Risk Group'; atRiskColor = '#c084fc';
    }

    // ── Combine risk factors (deduplicated top-5) ──────────────────────────
    const allRiskFactors = [
      ...(occData?.riskFactors || []),
      ...(ageData?.riskFactors || []),
      ...(genData?.riskFactors || []),
      ...(areaData?.riskFactors || [])
    ].slice(0, 5);

    const allProtective = [
      ...(occData?.protective || []),
      ...(ageData?.protective || []),
      ...(genData?.protective || []),
      ...(areaData?.protective || [])
    ].slice(0, 4);

    // ── Primary insight sentence ────────────────────────────────────────────
    let insight = '';
    if (occData) {
      insight = occData.note || '';
    } else if (ageData) {
      insight = ageData.note || '';
    }
    // Append gender insight if meaningful
    if (genData?.insight && gender !== 'prefer_not') {
      insight = insight ? insight + ' ' + genData.insight : genData.insight;
    }
    // Append area insight
    if (areaData?.insight) {
      insight = insight ? insight + ' ' + areaData.insight : areaData.insight;
    }

    // ── Comparison to national average ─────────────────────────────────────
    const nationalAvg = 14.3; // NMHS 2016 overall prevalence
    const ratio = (prevalence / nationalAvg).toFixed(1);
    const comparisonText = prevalence > nationalAvg + 3
      ? `People in your demographic group have ~${ratio}× the national average mental health risk (India, NMHS 2016).`
      : prevalence < nationalAvg - 3
        ? `Your demographic group has a relatively lower reported mental health risk than the national average — though access to care is the key variable.`
        : `Your demographic group is near the national average prevalence rate for mental health concerns in India.`;

    // ── Profile tags for UI ─────────────────────────────────────────────────
    const tags = [];
    if (ageData)  tags.push({ label: ageData.label,  color: '#c084fc' });
    if (genData && gender !== 'prefer_not')  tags.push({ label: genData.label,  color: '#f472b6' });
    if (occData)  tags.push({ label: occData.emoji + ' ' + occData.label, color: '#06d6a0' });
    if (areaData) tags.push({ label: areaData.label, color: '#22d3ee' });

    return {
      prevalence,
      multiplier,
      atRiskLabel,
      atRiskColor,
      tags,
      riskFactors:   allRiskFactors,
      protective:    allProtective,
      comparisonText,
      insight:       insight.slice(0, 420),  // cap for UI
      ageData,
      genData,
      occData,
      areaData,
      hasGender:  !!genData && gender !== 'prefer_not',
      hasAge:     !!ageData,
      hasOcc:     !!occData,
      hasArea:    !!areaData
    };
  }

  // ── Save / load profile from localStorage ────────────────────────────────
  function saveProfile(profile) {
    try {
      localStorage.setItem('mb_demo_profile', JSON.stringify(profile));
    } catch (e) { /* storage full or private browsing */ }
  }

  function loadProfile() {
    try {
      const raw = localStorage.getItem('mb_demo_profile');
      return raw ? JSON.parse(raw) : {};
    } catch (e) { return {}; }
  }

  // ── Expose data objects for UI building ────────────────────────────────────
  function getAgeOptions() {
    return Object.entries(AGE_DATA).map(([k, v]) => ({ value: k, label: v.label }));
  }
  function getOccupationOptions() {
    return Object.entries(OCCUPATION_DATA).map(([k, v]) => ({ value: k, label: v.emoji + ' ' + v.label }));
  }

  return {
    analyze,
    saveProfile,
    loadProfile,
    getAgeOptions,
    getOccupationOptions,
    AGE_DATA,
    GENDER_DATA,
    OCCUPATION_DATA,
    AREA_DATA
  };

})();
