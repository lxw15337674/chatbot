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

    setIsSuccessful(true);
    toast({ type: "success", description: "Signed in locally." });
    router.push("/");
  };

  return (
    <>
      <h1 className="text-2xl font-semibold tracking-tight">Welcome back</h1>
      <p className="text-sm text-muted-foreground">
        Sign in to your account to continue
      </p>
      <AuthForm action={handleSubmit} defaultEmail={email}>
        <SubmitButton isSuccessful={isSuccessful}>Sign in</SubmitButton>
        <p className="text-center text-[13px] text-muted-foreground">
          {"No account? "}
          <Link
            className="text-foreground underline-offset-4 hover:underline"
            href="/register"
          >
            Sign up
          </Link>
        </p>
      </AuthForm>
    </>
  );
}
