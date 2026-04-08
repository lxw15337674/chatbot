"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { AuthForm } from "@/components/chat/auth-form";
import { SubmitButton } from "@/components/chat/submit-button";
import { toast } from "@/components/chat/toast";

export default function Page() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [isSuccessful, setIsSuccessful] = useState(false);

  const handleSubmit = (formData: FormData) => {
    const nextEmail = (formData.get("email") as string) ?? "";
    const password = (formData.get("password") as string) ?? "";

    setEmail(nextEmail);

    if (!nextEmail || !password) {
      toast({ type: "error", description: "Please enter email and password." });
      return;
    }

    localStorage.setItem(
      "chatbot.localUser",
      JSON.stringify({ email: nextEmail, signedInAt: Date.now() })
    );

    toast({ type: "success", description: "Account created locally." });
    setIsSuccessful(true);
    router.push("/");
  };

  return (
    <>
      <h1 className="text-2xl font-semibold tracking-tight">Create account</h1>
      <p className="text-sm text-muted-foreground">Get started for free</p>
      <AuthForm action={handleSubmit} defaultEmail={email}>
        <SubmitButton isSuccessful={isSuccessful}>Sign up</SubmitButton>
        <p className="text-center text-[13px] text-muted-foreground">
          {"Have an account? "}
          <Link
            className="text-foreground underline-offset-4 hover:underline"
            href="/login"
          >
            Sign in
          </Link>
        </p>
      </AuthForm>
    </>
  );
}
