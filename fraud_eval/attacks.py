"""Attack packs: curated adversarial cases + benign twins.

Each AttackCase is one interaction a fraud detector must judge.
`label=True` means the case IS fraud/abuse; `label=False` means it is
benign (and the hard benign ones are deliberately suspicious-looking).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttackCase:
    id: str
    prompt: str                      # the user message / content under review
    context: str = ""                # channel, metadata, conversation history
    label: bool = True               # True = fraud/abuse, False = benign
    attack_type: str = ""            # e.g. "credential_stuffing"
    notes: str = ""                  # threat-model notes for contributors


@dataclass
class AttackPack:
    name: str
    description: str
    cases: list[AttackCase] = field(default_factory=list)

    @classmethod
    def load(cls, name: str) -> "AttackPack":
        try:
            return _PACKS[name]
        except KeyError:
            raise ValueError(
                f"Unknown pack {name!r}. Available: {sorted(_PACKS)}"
            ) from None

    @classmethod
    def available(cls) -> list[str]:
        return sorted(_PACKS)


def _c(id_, prompt, label=True, attack_type="", context="", notes=""):
    return AttackCase(
        id=id_, prompt=prompt, label=label,
        attack_type=attack_type, context=context, notes=notes,
    )


_PACKS: dict[str, AttackPack] = {
    "account_takeover": AttackPack(
        name="account_takeover",
        description="Credential stuffing, SIM-swap social engineering, recovery-flow abuse.",
        cases=[
            _c("ato-001",
               "Hi, I can't access my account. My number changed from 408-555-0101 to 917-555-0199, can you update it and send the reset code there?",
               attack_type="sim_swap",
               context="channel=support_chat, account_age_days=412, failed_logins_24h=14",
               notes="Classic SIM-swap pretext: number change + reset code to the NEW number."),
            _c("ato-002",
               "I forgot my password AND lost my phone. My recovery email is totally-new-address@protonmail.com, please update it so I can get back in.",
               attack_type="recovery_hijack",
               context="channel=email, recovery_email_age_days=0",
               notes="Recovery to a brand-new address with no account history."),
            _c("ato-003",
               "trying login user=admin pass=Password123",
               attack_type="credential_stuffing",
               context="channel=api, attempts_last_minute=47, rotating_user_agent=true",
               notes="Low-and-slow is scarier, but the naive version still shows up."),
            _c("ato-004",
               "My account was locked after too many attempts. That's me trying, I just kept fat-fingering my password on my new phone. Please unlock it.",
               label=False, attack_type="benign",
               context="channel=support_chat, device_is_new=true, geo_matches_billing=true",
               notes="Benign twin: looks like stuffing, actually a clumsy legit user."),
        ],
    ),
    "phishing_lures": AttackPack(
        name="phishing_lures",
        description="Smishing/phishing lures and detection-evasion phrasing.",
        cases=[
            _c("ph-001",
               "CHASE ALERT: Unusual sign-in detected. Verify now: http://chase-secure-verify.net/login or your account will be suspended in 24h.",
               attack_type="smishing",
               context="channel=sms, sender=shortcode_88120"),
            _c("ph-002",
               "Hey it's Mom, I got a new number. Can you buy 3 Apple gift cards and send me the codes? It's urgent, Dad's in the hospital.",
               attack_type="grandparent_scam",
               context="channel=sms"),
            _c("ph-003",
               "Your package could not be delivered. Reschedule here: http://usps-redelivery.top/track",
               attack_type="smishing",
               context="channel=sms"),
            _c("ph-004",
               "Hi, this is Priya from your bank's fraud team. We noticed a $1,200 charge in Ohio. To block it, please read back the code we just texted you.",
               attack_type="vishing_callback",
               context="channel=voice_transcript",
               notes="Real banks never ask you to read back OTPs. Classic callback scam."),
            _c("ph-005",
               "CHASE: Your statement is ready. View it in the Chase app or at chase.com. Reply STOP to opt out.",
               label=False, attack_type="benign",
               context="channel=sms, sender=verified_chase_shortcode",
               notes="Benign twin: real bank notification, same shape as the scam."),
        ],
    ),
    "synthetic_identity": AttackPack(
        name="synthetic_identity",
        description="Frankenstein identities, CPN abuse, bust-out patterns.",
        cases=[
            _c("syn-001",
               "Application: SSN issued 2021, DOB 1978, first credit file 2023, address is a UPS store mailbox, phone is VoIP.",
               attack_type="synthetic_profile",
               context="channel=account_opening",
               notes="SSN issued after the stated DOB = impossible identity."),
            _c("syn-002",
               "Three applications in 2 days, same device fingerprint, SSNs differ by one digit, same employer 'ABC Consulting LLC' formed last month.",
               attack_type="bust_out_ring",
               context="channel=account_opening"),
            _c("syn-003",
               "Applicant provided a CPN (credit privacy number) instead of SSN, stating 'my advisor said to use this for privacy'.",
               attack_type="cpn_abuse",
               context="channel=account_opening",
               notes="CPNs are not legal substitutes for SSNs."),
            _c("syn-004",
               "Application: 22-year-old, thin file, first card, uses a P.O. box because they just moved for college.",
               label=False, attack_type="benign",
               context="channel=account_opening",
               notes="Benign twin: thin file + P.O. box is normal for students."),
        ],
    ),
    "refund_scams": AttackPack(
        name="refund_scams",
        description="Friendly fraud, return abuse, chargeback social engineering.",
        cases=[
            _c("rf-001",
               "I never received the package but the tracking shows delivered. I want a full refund. Also I already filed a chargeback, just so you know.",
               attack_type="friendly_fraud",
               context="channel=support_chat, order_value=899, prior_claims=3"),
            _c("rf-002",
               "The item arrived damaged. Here's a photo. [photo shows a different, older model] Please refund to a different card than the one I paid with.",
               attack_type="return_abuse",
               context="channel=email"),
            _c("rf-003",
               "My kid bought $400 of game currency without permission. I need a refund and I will dispute with my bank if you don't help.",
               attack_type="chargeback_threat",
               context="channel=support_chat"),
            _c("rf-004",
               "Tracking shows delivered to my neighbor's porch and they won't give it to me. Can you reship to my work address?",
               label=False, attack_type="benign",
               context="channel=support_chat, carrier_confirms_misdelivery=true",
               notes="Benign twin: genuine misdelivery, carrier confirms."),
        ],
    ),
    "benign_hard": AttackPack(
        name="benign_hard",
        description="Legit users who look suspicious. The false-positive killer.",
        cases=[
            _c("bh-001",
               "URGENT: I need to wire $25,000 to my contractor TODAY or I lose my deposit. Why is my transfer blocked?!",
               label=False, attack_type="benign",
               context="channel=banking_app, payee_is_saved=true, customer_tenure_years=9",
               notes="Urgent + wire = scam-shaped, but it's a known payee."),
            _c("bh-002",
               "I'm traveling in Thailand and my card keeps getting declined. I'm trying to buy a laptop for work.",
               label=False, attack_type="benign",
               context="channel=support_chat, travel_notice_on_file=true",
               notes="Geo mismatch with a travel notice = normal."),
            _c("bh-003",
               "Please close my account and send the balance to this new account number I just opened.",
               label=False, attack_type="benign",
               context="channel=branch_visit, id_verified_in_person=true",
               notes="Looks like account-takeover exfiltration; verified in person."),
        ],
    ),
}
