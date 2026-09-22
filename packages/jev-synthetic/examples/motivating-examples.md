# Human-authored task designs, not generated results

These illustrative inputs were written for this project and are not quotations from HF sources. A teacher must find comparable evidence in its actual source or reject the task.

| Domain | Input | Primitive and question | Options / label |
|---|---|---|---|
| Financial | Revenue rose from 80 to 100 million. Operating costs remained at 60 million. | Choice: Which measure increased? | Revenue; operating costs. Index 0. |
| Legal | The supplier may terminate only after giving thirty days' written notice. | Noul: May the supplier terminate without written notice? | false; true. Index 0. Explicit contradiction, not missing evidence. |
| Everyday | The appointment is Tuesday at 10. The patient asks to move it to Wednesday. The receptionist confirms Wednesday at 11. | Choice: What is the confirmed appointment? | Tuesday 10; Wednesday 10; Wednesday 11. Index 2. |
| Financial | Two out of four listed milestones were completed. | Score: Rate milestone completion: 0 = none, 1 = some but not all, 2 = all. | None; some; all. Index 1; numeric values 0,1,2. |
| General | The observatory opened in 1910. Its second telescope arrived in 1932. | Noul: Did the second telescope arrive after the observatory opened? | false; true. Index 1. |
| Legal | The agreement requires notification within five business days. | Choice: Which time unit governs the notification deadline? | Calendar days; business days; weeks. Index 1. |

Use functions such as entity/relation extraction, time ordering, contradiction, numerical comparison, policy application and explicit rubric classification. A varied domain alone does not guarantee varied reasoning. Avoid requiring legal/financial world knowledge beyond the provided text, and never turn missing evidence into a negative answer.
